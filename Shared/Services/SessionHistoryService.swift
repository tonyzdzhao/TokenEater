import Foundation

/// JSONL aggregator backing the History view. Designed to scale to hundreds of
/// session files without freezing the UI:
/// - mtime filter -> only scan files that could possibly contain data in the
///   requested range
/// - per-file persisted cache keyed by path + mtime -> repeated opens after
///   the first cold scan re-use the previous result and only re-parse the
///   handful of files that changed since
/// - append-aware re-parse -> a transcript whose mtime moved is read only
///   from where the previous parse stopped (see `parseClaudeTranscript`)
/// - chunked byte streaming + `input_tokens` pre-filter -> JSON parse only the
///   assistant turns that actually carry token usage
/// - TaskGroup with concurrency cap -> uses cores without saturating
/// - cooperative cancellation via `Task.checkCancellation()`
final class SessionHistoryService: SessionHistoryServiceProtocol {

    /// Cache file path. Same dir as the shared usage file so it's easy
    /// to nuke during dev. Falls back to `NSHomeDirectory()` when the
    /// user domain isn't fully resolved (some CI / sandboxed envs).
    private static var cacheURL: URL {
        let support = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first
            ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return support
            .appendingPathComponent("com.tokeneater.shared", isDirectory: true)
            .appendingPathComponent("history-cache.json")
    }

    /// Root scan dir. Real home, not the sandbox container -> we resolve via
    /// `getpwuid` (same trick used in `SharedFileService` for the widget).
    private static var projectsURL: URL {
        let pw = getpwuid(getuid())
        let home: URL
        if let home_str = pw?.pointee.pw_dir.flatMap({ String(cString: $0) }), !home_str.isEmpty {
            home = URL(fileURLWithPath: home_str)
        } else {
            home = FileManager.default.homeDirectoryForCurrentUser
        }
        return home.appendingPathComponent(".claude/projects", isDirectory: true)
    }

    /// 4 files in parallel keeps the work tight without saturating the IO
    /// queue or starving the rest of the app of cores.
    private static let concurrencyCap = 4

    enum Source: Sendable {
        case claude
        case codex

        var provider: MetricProvider {
            switch self {
            case .claude: return .claude
            case .codex:  return .codex
            }
        }
    }

    private let source: Source
    private let roots: [URL]
    private let persistedCacheURL: URL

    init(source: Source = .claude, rootURL: URL? = nil, cacheURL: URL? = nil) {
        self.source = source
        switch source {
        case .claude:
            roots = [rootURL ?? Self.projectsURL]
            persistedCacheURL = cacheURL ?? Self.cacheURL
        case .codex:
            let home = rootURL ?? CodexAuthReader().codexHomeURL
            roots = [home.appendingPathComponent("sessions"), home.appendingPathComponent("archived_sessions")]
            persistedCacheURL = cacheURL ?? Self.cacheURL.deletingLastPathComponent()
                .appendingPathComponent("codex-history-cache.json")
        }
    }

    // MARK: - Public

    func loadHistory(range: HistoryRange) async throws -> [HistoryBucket] {
        try await loadAggregates(rangeStart: rangeStart(for: range), bucketing: range)
    }

    func loadPreviousPeriodActiveTokens(range: HistoryRange) async throws -> [MetricProvider: Int] {
        let now = Date()
        let currentStart = range == .sevenDays
            ? ChartDomainCalculator.domain(range: range, now: now).start
            : now.addingTimeInterval(-range.seconds)
        // Both windows must span the same duration or the delta reports a
        // decline that is only a difference in length. The 7-day domain starts
        // at midnight six days ago, so the current window runs 6.0 to 7.0 days
        // depending on the time of day; a flat -7d previous window made the
        // hero card read about -14% at 00:30 on perfectly steady usage.
        let previousStart = range == .sevenDays
            ? currentStart.addingTimeInterval(-now.timeIntervalSince(currentStart))
            : currentStart.addingTimeInterval(-range.seconds)
        let buckets = try await loadAggregates(
            rangeStart: previousStart,
            rangeEnd: currentStart,
            bucketing: range
        )
        return [source.provider: buckets.reduce(0) { $0 + $1.totalActive }]
    }

    // MARK: - Aggregation pipeline

    private func loadAggregates(
        rangeStart: Date,
        rangeEnd: Date = Date(),
        bucketing: HistoryRange
    ) async throws -> [HistoryBucket] {
        // 1. List candidate files via FileManager + mtime filter.
        let files = try candidateFiles(rangeStart: rangeStart)
        try Task.checkCancellation()

        // 2. Load existing cache off the main queue (small JSON, cheap).
        let cache = loadCache()
        let source = self.source

        // 3. Parse each candidate file in a TaskGroup, hitting the cache when
        //    mtime matches. We collect cache entries so we can persist them
        //    back at the end of the run.
        var newEntries: [String: HistoryFileCacheEntry] = [:]
        try await withThrowingTaskGroup(of: HistoryFileCacheEntry?.self) { group in
            var inFlight = 0
            var iterator = files.makeIterator()

            // Prime the pump up to the concurrency cap.
            while inFlight < Self.concurrencyCap, let url = iterator.next() {
                group.addTask { try await Self.parseOrReuse(url: url, cache: cache, source: source) }
                inFlight += 1
            }

            while let result = try await group.next() {
                inFlight -= 1
                if let entry = result {
                    newEntries[entry.path] = entry
                }
                if let url = iterator.next() {
                    group.addTask { try await Self.parseOrReuse(url: url, cache: cache, source: source) }
                    inFlight += 1
                }
                try Task.checkCancellation()
            }
        }

        // 4. Merge new entries back into the cache and persist.
        //
        //    This used to be `cache.entries = newEntries`, which looked tidy
        //    and quietly evicted every file the current range could not see.
        //    The mtime filter in step 1 drops anything older than rangeStart,
        //    so a 7-day scan rewrote the cache with only the last week's
        //    files and a following 30-day scan re-parsed everything from
        //    scratch. Since `MonitoringInsightsStore` (current window plus
        //    previous window), `HistoryWidgetStore` and the History view all
        //    scan with different ranges, the cache spent its life being
        //    rebuilt and the expensive cold parse never stopped happening.
        //
        //    Re-read immediately before writing rather than reusing the copy
        //    taken in step 2: several scans run concurrently, and the shared
        //    file next door needs the same read-modify-write discipline for
        //    the same reason. Then merge, and prune only what has actually
        //    disappeared from disk, so the cache still cannot grow forever.
        var persisted = loadCache()
        persisted.entries.merge(newEntries) { _, fresh in fresh }
        persisted.entries = persisted.entries.filter {
            FileManager.default.fileExists(atPath: $0.key)
        }
        saveCache(persisted)

        // 5. Aggregate across files into the requested bucketing.
        return Self.aggregate(
            entries: deduplicatedEntries(Array(newEntries.values)),
            rangeStart: rangeStart,
            rangeEnd: rangeEnd,
            bucketing: bucketing
        )
    }

    // MARK: - File discovery

    private func candidateFiles(rangeStart: Date) throws -> [URL] {
        let fm = FileManager.default
        var results: [URL] = []

        for root in roots {
            guard let enumerator = fm.enumerator(
                at: root,
                includingPropertiesForKeys: [.contentModificationDateKey, .isRegularFileKey],
                options: [.skipsHiddenFiles]
            ) else {
                continue
            }

            for case let url as URL in enumerator {
                guard url.pathExtension == "jsonl" else { continue }
                let attrs = try? url.resourceValues(forKeys: [.contentModificationDateKey, .isRegularFileKey])
                guard attrs?.isRegularFile == true else { continue }
                // mtime filter: a file with mtime older than rangeStart cannot
                // possibly contain events relevant to the requested window. Skip.
                if let mtime = attrs?.contentModificationDate, mtime < rangeStart {
                    continue
                }
                results.append(url)
            }
        }
        return results
    }

    // MARK: - Per-file parsing

    private static func parseOrReuse(
        url: URL,
        cache: HistoryCache,
        source: Source
    ) async throws -> HistoryFileCacheEntry? {
        try Task.checkCancellation()

        let attrs = try? url.resourceValues(forKeys: [.contentModificationDateKey])
        guard let mtime = attrs?.contentModificationDate else { return nil }

        // Cache hit: same mtime -> reuse without touching disk.
        let cached = cache.entries[url.path]
        if let cached, cached.mtime == mtime {
            return cached
        }

        if source == .codex {
            let content: String
            do {
                content = try String(contentsOf: url, encoding: .utf8)
            } catch {
                return nil
            }
            return try CodexHistoryParser.parse(content, path: url.path, mtime: mtime)
        }

        return try parseClaudeTranscript(url: url, mtime: mtime, previous: cached)
    }

    /// Bytes per read. Bounds memory on the 500 MB+ transcripts that long
    /// sessions reach, which a whole-file `String(contentsOf:)` loaded and
    /// UTF-8-validated in one piece.
    private static let readChunkSize = 4 << 20

    /// Parse a Claude transcript into hourly buckets, resuming from the
    /// previous entry when the file has only grown since.
    ///
    /// A live session's transcript gets a new mtime on every turn, so a
    /// whole-file parse re-read every active transcript from byte 0 on every
    /// scan. With ~20 open sessions that was over 1 GB of JSONL every 5
    /// minutes (the `HistoryWidgetStore` cadence), two cores pegged for
    /// seconds each time. Claude Code only appends to a transcript, so the
    /// previous buckets plus the lines after `parsedBytes` are exactly what a
    /// full parse produces. A file that shrank or was replaced (another inode)
    /// is parsed from scratch.
    ///
    /// Only whole lines are consumed. A last line without its newline is
    /// usually one Claude Code is still writing, so it waits for the next scan,
    /// unless it already parses as a complete JSON object (a finished file
    /// whose last line just lacks the newline).
    private static func parseClaudeTranscript(
        url: URL,
        mtime: Date,
        previous: HistoryFileCacheEntry?
    ) throws -> HistoryFileCacheEntry? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }

        var info = stat()
        guard fstat(handle.fileDescriptor, &info) == 0 else { return nil }
        let size = Int64(info.st_size)
        let inode = UInt64(info.st_ino)

        var acc = ClaudeTranscriptAccumulator(fallbackProject: decodedProject(from: url))
        var offset: Int64 = 0
        if let previous,
           let parsed = previous.parsedBytes,
           previous.inode == inode,
           parsed <= size {
            acc.seed(with: previous)
            offset = parsed
            if offset > 0 { try handle.seek(toOffset: UInt64(offset)) }
        }

        var pending = Data()
        while true {
            if Task.isCancelled { throw CancellationError() }
            guard let chunk = try handle.read(upToCount: readChunkSize), !chunk.isEmpty else { break }
            pending.append(chunk)
            let consumed = acc.consumeCompleteLines(in: pending)
            if consumed > 0 {
                pending = Data(pending.dropFirst(consumed))
                offset += Int64(consumed)
            }
        }
        if !pending.isEmpty, acc.consumeTrailingLine(pending) {
            offset += Int64(pending.count)
        }

        return acc.entry(path: url.path, mtime: mtime, parsedBytes: offset, inode: inode)
    }

    /// Running aggregation of one Claude transcript, fed line by line.
    private struct ClaudeTranscriptAccumulator {
        /// Cheap pre-filter: only lines carrying token usage are JSON-parsed,
        /// which skips user prompts, tool results, system events and the like.
        private static let usageMarker = Array("input_tokens".utf8)

        let fallbackProject: String
        private var bucketsByHour: [Date: HistoryBucket] = [:]
        private var sessionIds: Set<String> = []

        init(fallbackProject: String) {
            self.fallbackProject = fallbackProject
        }

        /// Start from a previous parse of the same file. Its session count
        /// sits on its earliest bucket; it is zeroed here and recomputed in
        /// `entry`, so it is never counted twice.
        mutating func seed(with entry: HistoryFileCacheEntry) {
            for var bucket in entry.buckets {
                bucket.sessionsCount = 0
                bucketsByHour[bucket.date] = bucket
            }
            sessionIds = Set(entry.sessionIds)
        }

        /// Feed every newline-terminated line in `data`. Returns the number of
        /// bytes consumed, which ends just after the last newline.
        mutating func consumeCompleteLines(in data: Data) -> Int {
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int in
                guard let base = raw.baseAddress else { return 0 }
                let count = raw.count
                var lineStart = 0
                while lineStart < count,
                      let newline = memchr(base + lineStart, 0x0A, count - lineStart) {
                    let lineEnd = base.distance(to: UnsafeRawPointer(newline))
                    consumeLine(base + lineStart, length: lineEnd - lineStart)
                    lineStart = lineEnd + 1
                }
                return lineStart
            }
        }

        /// Feed a last line that has no newline yet. Returns true when it was
        /// a complete JSON object (and so is consumed for good), false when it
        /// looks partial and must be read again on the next scan.
        mutating func consumeTrailingLine(_ data: Data) -> Bool {
            guard let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
                return false
            }
            let hasUsage = data.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return false }
                return Self.containsUsageMarker(base, length: raw.count)
            }
            if hasUsage { add(obj) }
            return true
        }

        private static func containsUsageMarker(_ start: UnsafeRawPointer, length: Int) -> Bool {
            usageMarker.withUnsafeBytes { marker in
                memmem(start, length, marker.baseAddress, marker.count) != nil
            }
        }

        private mutating func consumeLine(_ start: UnsafeRawPointer, length: Int) {
            guard length > 0, Self.containsUsageMarker(start, length: length) else { return }
            let line = Data(bytes: start, count: length)
            guard let obj = (try? JSONSerialization.jsonObject(with: line)) as? [String: Any] else { return }
            add(obj)
        }

        private mutating func add(_ obj: [String: Any]) {
            // Timestamp -> bucket date (start of hour).
            guard let tsString = obj["timestamp"] as? String,
                  let date = SessionHistoryService.isoDate(tsString) else { return }
            let bucketDate = SessionHistoryService.startOfHour(date)

            // Session id -> contributes to the distinct count.
            if let sessionId = obj["sessionId"] as? String {
                sessionIds.insert(sessionId)
            }

            // Project path: prefer the per-event `cwd` field. The
            // directory-name decoding fallback is lossy for project paths
            // that contain hyphens (`/` -> `-` collapses `/a/b-c` and
            // `/a/b/c` to the same key).
            let projectPath: String
            if let cwd = obj["cwd"] as? String, !cwd.isEmpty {
                projectPath = cwd
            } else {
                projectPath = fallbackProject
            }

            // Pull message + usage payload.
            guard let message = obj["message"] as? [String: Any],
                  let usage = message["usage"] as? [String: Any] else { return }

            let modelString = (message["model"] as? String) ?? ""
            let kind = ModelKind(rawModel: modelString)

            let input = (usage["input_tokens"] as? Int) ?? 0
            let output = (usage["output_tokens"] as? Int) ?? 0
            let cacheRead = (usage["cache_read_input_tokens"] as? Int) ?? 0
            let cacheCreate = (usage["cache_creation_input_tokens"] as? Int) ?? 0
            let active = input + cacheCreate + output

            var bucket = bucketsByHour[bucketDate] ?? HistoryBucket(
                date: bucketDate,
                tokensByModel: [:],
                tokensByProject: [:],
                sessionsCount: 0,
                inputTokens: 0,
                outputTokens: 0,
                cacheReadTokens: 0,
                cacheCreateTokens: 0
            )
            bucket.tokensByModel[kind, default: 0] += active
            bucket.tokensByProject[projectPath, default: 0] += active
            bucket.inputTokens += input
            bucket.outputTokens += output
            bucket.cacheReadTokens += cacheRead
            bucket.cacheCreateTokens += cacheCreate
            bucketsByHour[bucketDate] = bucket
        }

        func entry(path: String, mtime: Date, parsedBytes: Int64, inode: UInt64) -> HistoryFileCacheEntry {
            // Each file is a single session. The session-count contribution is
            // 1 per *distinct* session id observed. In practice that's always 1
            // for a JSONL file, but the format technically allows multiple ids
            // (resumes), so count properly. It goes on the earliest bucket so
            // the daily aggregator can sum without double-counting.
            var output = bucketsByHour
            if let earliest = output.keys.min() {
                output[earliest]?.sessionsCount = max(1, sessionIds.count)
            }
            return HistoryFileCacheEntry(
                path: path,
                mtime: mtime,
                buckets: Array(output.values).sorted { $0.date < $1.date },
                sessionIds: Array(sessionIds),
                parsedBytes: parsedBytes,
                inode: inode
            )
        }
    }

    // MARK: - Aggregation across files

    private static func aggregate(
        entries: [HistoryFileCacheEntry],
        rangeStart: Date,
        rangeEnd: Date,
        bucketing: HistoryRange
    ) -> [HistoryBucket] {
        var combined: [Date: HistoryBucket] = [:]

        for entry in entries {
            for hourly in entry.buckets {
                guard hourly.date >= rangeStart, hourly.date < rangeEnd else { continue }
                let key = bucketing.isHourly ? hourly.date : startOfDay(hourly.date)
                let existing = combined[key] ?? HistoryBucket(
                    date: key,
                    tokensByModel: [:],
                    tokensByProject: [:],
                    sessionsCount: 0,
                    inputTokens: 0,
                    outputTokens: 0,
                    cacheReadTokens: 0,
                    cacheCreateTokens: 0
                )
                combined[key] = HistoryBucket.merging(existing, hourly, date: key)
            }
        }

        return combined.values.sorted { $0.date < $1.date }
    }

    private func deduplicatedEntries(_ entries: [HistoryFileCacheEntry]) -> [HistoryFileCacheEntry] {
        guard source == .codex else { return entries }
        var sessions: [String: HistoryFileCacheEntry] = [:]
        for entry in entries.sorted(by: { $0.mtime < $1.mtime }) {
            sessions[entry.sessionIds.first ?? entry.path] = entry
        }
        return Array(sessions.values)
    }

    // MARK: - Cache persistence

    private func loadCache() -> HistoryCache {
        guard let data = try? Data(contentsOf: persistedCacheURL),
              let cache = try? JSONDecoder().decode(HistoryCache.self, from: data),
              cache.isCurrentVersion
        else {
            return .empty
        }
        return cache
    }

    private func saveCache(_ cache: HistoryCache) {
        let url = persistedCacheURL
        let dir = url.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        guard let data = try? JSONEncoder().encode(cache) else { return }
        try? data.write(to: url, options: .atomic)
    }

    // MARK: - Date helpers

    private static let isoFractional: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    private static let isoPlain: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    private static func isoDate(_ s: String) -> Date? {
        isoFractional.date(from: s) ?? isoPlain.date(from: s)
    }

    private static func startOfHour(_ date: Date) -> Date {
        let comps = Calendar.current.dateComponents([.year, .month, .day, .hour], from: date)
        return Calendar.current.date(from: comps) ?? date
    }

    private static func startOfDay(_ date: Date) -> Date {
        Calendar.current.startOfDay(for: date)
    }

    private func rangeStart(for range: HistoryRange) -> Date {
        range == .sevenDays
            ? ChartDomainCalculator.domain(range: range).start
            : Date().addingTimeInterval(-range.seconds)
    }

    // MARK: - Project path decode

    /// Claude Code stores sessions under a directory whose name is the project
    /// path with `/` swapped for `-` (e.g. `/Users/foo/repo` becomes
    /// `-Users-foo-repo`). We re-form the path for display by reversing that
    /// substitution. Fallback to the raw directory name if anything fails.
    private static func decodedProject(from fileURL: URL) -> String {
        let dir = fileURL.deletingLastPathComponent().lastPathComponent
        guard dir.hasPrefix("-") else { return dir }
        let withSlashes = "/" + String(dir.dropFirst()).replacingOccurrences(of: "-", with: "/")
        return withSlashes
    }
}
