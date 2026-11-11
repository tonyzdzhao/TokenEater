import Testing
import Foundation

/// A live transcript's mtime moves on every turn, so the history scan used to
/// re-read every open session's JSONL from byte 0 each time it ran. These tests
/// pin the append-aware parse that replaced it: resuming from the previous
/// offset must give exactly what a fresh full parse gives, a half-written last
/// line must be counted once (when it is finished), and a file that was
/// replaced rather than appended to must not be stacked on its old totals.
@Suite("Session history incremental parse")
struct SessionHistoryIncrementalTests {

    private func makeRoot() throws -> URL {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("history-incr-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private static let iso: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    /// One assistant turn with usage, `hoursAgo` before now so it always lands
    /// inside the 7-day window.
    private func usageLine(hoursAgo: Double, input: Int, output: Int = 1, session: String = "s-1") -> String {
        let ts = Self.iso.string(from: Date().addingTimeInterval(-hoursAgo * 3600))
        return #"{"type":"assistant","sessionId":"\#(session)","cwd":"/Users/test/proj","timestamp":"\#(ts)","message":{"role":"assistant","model":"claude-opus-5-5","usage":{"input_tokens":\#(input),"output_tokens":\#(output),"cache_read_input_tokens":7,"cache_creation_input_tokens":3}}}"#
    }

    private func userLine(hoursAgo: Double) -> String {
        let ts = Self.iso.string(from: Date().addingTimeInterval(-hoursAgo * 3600))
        return #"{"type":"user","sessionId":"s-1","cwd":"/Users/test/proj","timestamp":"\#(ts)","message":{"role":"user","content":"hello"}}"#
    }

    private func append(_ text: String, to url: URL) throws {
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(text.utf8))
    }

    private struct Totals: Equatable {
        var input = 0, output = 0, cacheRead = 0, cacheCreate = 0, sessions = 0
    }

    private func totals(_ buckets: [HistoryBucket]) -> Totals {
        buckets.reduce(into: Totals()) {
            $0.input += $1.inputTokens
            $0.output += $1.outputTokens
            $0.cacheRead += $1.cacheReadTokens
            $0.cacheCreate += $1.cacheCreateTokens
            $0.sessions += $1.sessionsCount
        }
    }

    /// What a cold scan of the same directory reports, with a cache of its own.
    private func freshTotals(root: URL) async throws -> Totals {
        let cacheURL = root.appendingPathComponent("fresh-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: cacheURL) }
        let service = SessionHistoryService(rootURL: root, cacheURL: cacheURL)
        return totals(try await service.loadHistory(range: .sevenDays))
    }

    private func cachedEntry(_ cacheURL: URL, named name: String) throws -> HistoryFileCacheEntry? {
        let cache = try JSONDecoder().decode(HistoryCache.self, from: Data(contentsOf: cacheURL))
        return cache.entries.first { ($0.key as NSString).lastPathComponent == name }?.value
    }

    @Test("Appending to a transcript resumes the parse and matches a full parse")
    func appendMatchesFullParse() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let cacheURL = root.appendingPathComponent("cache.json")
        let file = root.appendingPathComponent("s-1.jsonl")

        try ([userLine(hoursAgo: 30), usageLine(hoursAgo: 30, input: 100), usageLine(hoursAgo: 29, input: 50)]
            .joined(separator: "\n") + "\n").write(to: file, atomically: true, encoding: .utf8)

        let service = SessionHistoryService(rootURL: root, cacheURL: cacheURL)
        let first = totals(try await service.loadHistory(range: .sevenDays))
        #expect(first.input == 150)
        #expect(first.sessions == 1)
        let size1 = try #require(try FileManager.default.attributesOfItem(atPath: file.path)[.size] as? Int)
        #expect(try cachedEntry(cacheURL, named: "s-1.jsonl")?.parsedBytes == Int64(size1))

        // A later turn, in a new hour bucket, from the same session.
        try append(usageLine(hoursAgo: 2, input: 25) + "\n" + userLine(hoursAgo: 1) + "\n", to: file)
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(5)], ofItemAtPath: file.path)

        let second = totals(try await service.loadHistory(range: .sevenDays))
        #expect(second.input == 175)
        // Still one session: the seeded count must not be added again.
        #expect(second.sessions == 1)
        #expect(second == (try await freshTotals(root: root)))
        let size2 = try #require(try FileManager.default.attributesOfItem(atPath: file.path)[.size] as? Int)
        #expect(try cachedEntry(cacheURL, named: "s-1.jsonl")?.parsedBytes == Int64(size2))
    }

    @Test("A half-written last line is left for the next scan and counted once")
    func partialLastLineCountedOnce() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let cacheURL = root.appendingPathComponent("cache.json")
        let file = root.appendingPathComponent("s-1.jsonl")

        let finished = usageLine(hoursAgo: 5, input: 10)
        let inFlight = usageLine(hoursAgo: 4, input: 40)
        let cut = inFlight.index(inFlight.startIndex, offsetBy: inFlight.count / 2)
        try (finished + "\n" + String(inFlight[..<cut])).write(to: file, atomically: true, encoding: .utf8)

        let service = SessionHistoryService(rootURL: root, cacheURL: cacheURL)
        #expect(totals(try await service.loadHistory(range: .sevenDays)).input == 10)
        let parsed = try #require(try cachedEntry(cacheURL, named: "s-1.jsonl")?.parsedBytes)
        #expect(parsed == Int64(Data((finished + "\n").utf8).count))

        // Claude Code finishes the line.
        try append(String(inFlight[cut...]) + "\n", to: file)
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(5)], ofItemAtPath: file.path)

        let after = totals(try await service.loadHistory(range: .sevenDays))
        #expect(after.input == 50)
        #expect(after == (try await freshTotals(root: root)))
    }

    @Test("A complete last line without a newline is counted, and not again after more lines")
    func completeLastLineWithoutNewline() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let cacheURL = root.appendingPathComponent("cache.json")
        let file = root.appendingPathComponent("s-1.jsonl")

        try (usageLine(hoursAgo: 6, input: 10) + "\n" + usageLine(hoursAgo: 5, input: 20))
            .write(to: file, atomically: true, encoding: .utf8)

        let service = SessionHistoryService(rootURL: root, cacheURL: cacheURL)
        #expect(totals(try await service.loadHistory(range: .sevenDays)).input == 30)

        try append("\n" + usageLine(hoursAgo: 1, input: 5) + "\n", to: file)
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(5)], ofItemAtPath: file.path)

        let after = totals(try await service.loadHistory(range: .sevenDays))
        #expect(after.input == 35)
        #expect(after == (try await freshTotals(root: root)))
    }

    @Test("A replaced transcript is parsed from scratch, not stacked on the old totals")
    func replacedFileIsReparsed() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let cacheURL = root.appendingPathComponent("cache.json")
        let file = root.appendingPathComponent("s-1.jsonl")

        try ([usageLine(hoursAgo: 8, input: 100), usageLine(hoursAgo: 7, input: 100)]
            .joined(separator: "\n") + "\n").write(to: file, atomically: true, encoding: .utf8)
        let service = SessionHistoryService(rootURL: root, cacheURL: cacheURL)
        #expect(totals(try await service.loadHistory(range: .sevenDays)).input == 200)

        // Atomic rewrite: new inode. Longer than before, so a size check alone
        // would wrongly treat it as an append.
        try ([usageLine(hoursAgo: 3, input: 1), usageLine(hoursAgo: 2, input: 2), usageLine(hoursAgo: 1, input: 3)]
            .joined(separator: "\n") + "\n").write(to: file, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(5)], ofItemAtPath: file.path)

        #expect(totals(try await service.loadHistory(range: .sevenDays)).input == 6)
    }

    @Test("A truncated transcript is parsed from scratch")
    func shrunkFileIsReparsed() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let cacheURL = root.appendingPathComponent("cache.json")
        let file = root.appendingPathComponent("s-1.jsonl")

        try ([usageLine(hoursAgo: 8, input: 100), usageLine(hoursAgo: 7, input: 100)]
            .joined(separator: "\n") + "\n").write(to: file, atomically: true, encoding: .utf8)
        let service = SessionHistoryService(rootURL: root, cacheURL: cacheURL)
        #expect(totals(try await service.loadHistory(range: .sevenDays)).input == 200)

        // In-place truncation keeps the inode; the size going down is the tell.
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: 0)
        try handle.write(contentsOf: Data((usageLine(hoursAgo: 1, input: 9) + "\n").utf8))
        try handle.close()
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(5)], ofItemAtPath: file.path)

        #expect(totals(try await service.loadHistory(range: .sevenDays)).input == 9)
    }

    @Test("Lines longer than one read chunk are still parsed whole")
    func lineSpanningChunks() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("s-1.jsonl")

        // A tool result bigger than the 4 MB read size, between two usage lines.
        let big = #"{"type":"user","sessionId":"s-1","message":{"content":""# + String(repeating: "x", count: 5 << 20) + #""}}"#
        try ([usageLine(hoursAgo: 3, input: 11), big, usageLine(hoursAgo: 2, input: 22)]
            .joined(separator: "\n") + "\n").write(to: file, atomically: true, encoding: .utf8)

        #expect(try await freshTotals(root: root).input == 33)
    }

    @Test("A cache entry written before parsedBytes existed still decodes")
    func legacyEntryDecodes() throws {
        let json = #"{"version":\#(HistoryCache.currentVersion),"entries":{"/a.jsonl":{"path":"/a.jsonl","mtime":0,"buckets":[],"sessionIds":[]}}}"#
        let cache = try JSONDecoder().decode(HistoryCache.self, from: Data(json.utf8))
        #expect(cache.entries["/a.jsonl"]?.parsedBytes == nil)
        #expect(cache.entries["/a.jsonl"]?.inode == nil)
    }
}
