//! Incremental JSON pushdown automaton for masked decoding.
//!
//! The decoder never re-parses its output: it carries this automaton, clones it per
//! candidate token, and keeps the candidates that leave the machine alive. That is what makes
//! format masking O(vocabulary) instead of O(document) per step.
//!
//! The machine is the streaming view of the language in [`crate::json`]. It deliberately does
//! *not* remember the set of object keys it has seen (a streaming recognizer cannot do that in
//! finite state); a document with duplicate keys is still rejected by [`crate::json::parse`],
//! which is the authoritative final check. Validity of prefixes matches
//! [`crate::json::prefix_status`], and the tests pin that equivalence.

/// Structural expectation for the next character.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Need {
    /// A value must start here (after `[`, `,` or `:`).
    Value,
    /// A value may start, or the just-opened container may close.
    ValueOrClose,
    /// A key must start here (after `,` in an object).
    Key,
    /// A key may start, or the just-opened object may close.
    KeyOrClose,
    Colon,
    Comma,
    /// A complete document; only whitespace may follow.
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frame {
    Array,
    Object,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Key,
    Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Mode {
    Structure,
    String { role: Role },
    Escape { role: Role },
    Unicode { role: Role, left: u8 },
    Number,
    Literal { word: &'static str, matched: usize },
    Dead,
}

/// Something the automaton did, in order. The decoder uses this to know whether it just
/// closed an object (and therefore what may follow a separator).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Open,
    Close,
    Key,
    Value,
    Separator,
}

#[derive(Clone, Debug)]
pub struct Pda {
    stack: Vec<Frame>,
    need: Need,
    mode: Mode,
    number: String,
    events: Vec<Event>,
}

impl Default for Pda {
    fn default() -> Pda {
        Pda::new()
    }
}

impl Pda {
    pub fn new() -> Pda {
        Pda {
            stack: Vec::new(),
            need: Need::Value,
            mode: Mode::Structure,
            number: String::new(),
            events: Vec::new(),
        }
    }

    pub fn events(&self) -> &[Event] {
        &self.events
    }

    pub fn last_event(&self) -> Option<Event> {
        self.events.last().copied()
    }

    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// `false` once the machine has seen something that can never become JSON.
    pub fn is_alive(&self) -> bool {
        self.mode != Mode::Dead
    }

    /// A finished, complete document (no pending number, no open container).
    pub fn is_complete(&self) -> bool {
        self.mode == Mode::Structure && self.need == Need::Stop
    }

    pub fn push_str(&mut self, text: &str) -> bool {
        for ch in text.chars() {
            if !self.step(ch) {
                return false;
            }
        }
        true
    }

    pub fn step(&mut self, ch: char) -> bool {
        match self.mode.clone() {
            Mode::String { role } => self.step_string(ch, role),
            Mode::Escape { role } => self.step_escape(ch, role),
            Mode::Unicode { role, left } => self.step_unicode(ch, role, left),
            Mode::Number => self.step_number(ch),
            Mode::Literal { word, matched } => self.step_literal(ch, word, matched),
            Mode::Structure => self.step_structure(ch),
            Mode::Dead => false,
        }
    }

    fn die(&mut self) -> bool {
        self.mode = Mode::Dead;
        false
    }

    fn step_structure(&mut self, ch: char) -> bool {
        if ch.is_ascii_whitespace() {
            return true;
        }
        match self.need {
            Need::Stop => self.die(),
            Need::Value | Need::ValueOrClose => {
                if self.need == Need::ValueOrClose {
                    match (ch, self.stack.last()) {
                        (']', Some(Frame::Array)) => return self.close_container(),
                        ('}', Some(Frame::Object)) => return self.close_container(),
                        _ => {}
                    }
                }
                self.start_value(ch)
            }
            Need::Key | Need::KeyOrClose => {
                if self.need == Need::KeyOrClose && ch == '}' {
                    return self.close_container();
                }
                if ch == '"' {
                    self.mode = Mode::String { role: Role::Key };
                    return true;
                }
                self.die()
            }
            Need::Colon => {
                if ch == ':' {
                    self.need = Need::Value;
                    true
                } else {
                    self.die()
                }
            }
            Need::Comma => match (ch, self.stack.last().copied()) {
                (']', Some(Frame::Array)) | ('}', Some(Frame::Object)) => self.close_container(),
                (',', Some(Frame::Array)) => {
                    self.need = Need::Value;
                    self.events.push(Event::Separator);
                    true
                }
                (',', Some(Frame::Object)) => {
                    self.need = Need::Key;
                    self.events.push(Event::Separator);
                    true
                }
                _ => self.die(),
            },
        }
    }

    fn start_value(&mut self, ch: char) -> bool {
        match ch {
            '"' => {
                self.mode = Mode::String { role: Role::Value };
                true
            }
            't' => {
                self.mode = Mode::Literal {
                    word: "true",
                    matched: 1,
                };
                true
            }
            'f' => {
                self.mode = Mode::Literal {
                    word: "false",
                    matched: 1,
                };
                true
            }
            'n' => {
                self.mode = Mode::Literal {
                    word: "null",
                    matched: 1,
                };
                true
            }
            '[' => {
                self.stack.push(Frame::Array);
                self.need = Need::ValueOrClose;
                self.events.push(Event::Open);
                true
            }
            '{' => {
                self.stack.push(Frame::Object);
                self.need = Need::KeyOrClose;
                self.events.push(Event::Open);
                true
            }
            '-' | '0'..='9' => {
                self.mode = Mode::Number;
                self.number.clear();
                self.number.push(ch);
                true
            }
            _ => self.die(),
        }
    }

    fn close_container(&mut self) -> bool {
        self.stack.pop();
        self.events.push(Event::Close);
        self.complete_value()
    }

    /// A value just ended (or a container closed); figure out what may follow.
    fn complete_value(&mut self) -> bool {
        self.mode = Mode::Structure;
        self.need = if self.stack.is_empty() {
            Need::Stop
        } else {
            Need::Comma
        };
        self.events.push(Event::Value);
        true
    }

    fn step_string(&mut self, ch: char, role: Role) -> bool {
        match ch {
            '"' => match role {
                Role::Key => {
                    self.mode = Mode::Structure;
                    self.need = Need::Colon;
                    self.events.push(Event::Key);
                    true
                }
                Role::Value => self.complete_value(),
            },
            '\\' => {
                self.mode = Mode::Escape { role };
                true
            }
            ch if (ch as u32) < 0x20 => self.die(),
            _ => true,
        }
    }

    fn step_escape(&mut self, ch: char, role: Role) -> bool {
        match ch {
            '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {
                self.mode = Mode::String { role };
                true
            }
            'u' => {
                self.mode = Mode::Unicode { role, left: 4 };
                true
            }
            _ => self.die(),
        }
    }

    fn step_unicode(&mut self, ch: char, role: Role, left: u8) -> bool {
        if !ch.is_ascii_hexdigit() {
            return self.die();
        }
        if left <= 1 {
            self.mode = Mode::String { role };
        } else {
            self.mode = Mode::Unicode { role, left: left - 1 };
        }
        true
    }

    fn step_number(&mut self, ch: char) -> bool {
        if ch.is_ascii_digit() || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E' {
            self.number.push(ch);
            if !number_prefix_ok(&self.number, false) {
                return self.die();
            }
            // A positive exponent is monotone: extra digits can only make an overflow
            // worse, so an overflowing literal is already dead (``72E934`` used to stay
            // alive here while the strict parser said invalid).  A negative exponent can
            // still shrink the value into range (``9…9e-4`` overflows, ``9…9e-400`` is
            // finite), so it must stay open.
            if has_exponent(&self.number)
                && number_prefix_ok(&self.number, true)
                && !exponent_can_shrink(&self.number)
                && !number_is_finite(&self.number)
            {
                return self.die();
            }
            return true;
        }
        // The number ends here: it must be complete, then this char is structural again.
        if !number_prefix_ok(&self.number, true) || !number_is_finite(&self.number) {
            return self.die();
        }
        self.number.clear();
        self.complete_value();
        self.step(ch)
    }

    fn step_literal(&mut self, ch: char, word: &'static str, matched: usize) -> bool {
        let bytes = word.as_bytes();
        if matched >= bytes.len() || ch != bytes[matched] as char {
            return self.die();
        }
        if matched + 1 == bytes.len() {
            self.complete_value()
        } else {
            self.mode = Mode::Literal {
                word,
                matched: matched + 1,
            };
            true
        }
    }

    /// Close a pending number and require a complete document; `None` means `self` cannot
    /// become one.
    pub fn finish(&self) -> Option<Pda> {
        let mut finished = self.clone();
        if finished.mode == Mode::Number {
            if !number_prefix_ok(&finished.number, true) || !number_is_finite(&finished.number) {
                return None;
            }
            finished.number.clear();
            finished.complete_value();
        }
        if finished.is_complete() {
            Some(finished)
        } else {
            None
        }
    }

    pub fn accepts(&self) -> bool {
        self.finish().is_some()
    }

    /// Could a token *starting* with this byte possibly keep the machine alive?
    ///
    /// This is the cheap half of the XGrammar-style partition: it only looks at the current
    /// `(Mode, Need)` and the first byte, so a caller can drop most of the vocabulary before
    /// cloning the automaton.  It is deliberately conservative — `true` only means "worth
    /// trying", and every allowed token is still replayed in full by the caller.
    pub fn first_byte_allowed(&self, byte: u8) -> bool {
        let ch = byte as char;
        match &self.mode {
            Mode::Dead => false,
            Mode::String { .. } => !(byte < 0x20 || byte == b'"' || byte == b'\\'),
            Mode::Escape { .. } => matches!(
                ch,
                '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u'
            ),
            Mode::Unicode { .. } => ch.is_ascii_hexdigit(),
            Mode::Literal { word, matched } => word.as_bytes().get(*matched) == Some(&byte),
            Mode::Number => {
                ch.is_ascii_digit()
                    || matches!(ch, '-' | '+' | '.' | 'e' | 'E' | ',' | ']' | '}')
                    || ch.is_ascii_whitespace()
            }
            Mode::Structure => {
                if ch.is_ascii_whitespace() {
                    // Whitespace is legal in front of every structural expectation.
                    return true;
                }
                match self.need {
                Need::Stop => false,
                Need::Colon => ch == ':',
                Need::Comma => {
                    ch == ','
                        || (ch == ']' && matches!(self.stack.last(), Some(Frame::Array)))
                        || (ch == '}' && matches!(self.stack.last(), Some(Frame::Object)))
                }
                Need::Key | Need::KeyOrClose => {
                    ch == '"' || (self.need == Need::KeyOrClose && ch == '}')
                }
                Need::Value | Need::ValueOrClose => {
                    matches!(ch, '"' | '{' | '[' | 't' | 'f' | 'n' | '-' | '0'..='9')
                        || (self.need == Need::ValueOrClose
                            && ((ch == ']' && matches!(self.stack.last(), Some(Frame::Array)))
                                || (ch == '}'
                                    && matches!(self.stack.last(), Some(Frame::Object)))))
                }
                }
            }
        }
    }

    /// Like [`Pda::candidates`], but skips tokens whose first byte cannot possibly advance the
    /// machine. On a real BPE vocabulary this removes ~90% of the clone-and-replay work.
    pub fn candidates_classified(&self, classes: &TokenClasses) -> Vec<u32> {
        let mut allowed = Vec::new();
        for (index, token) in classes.tokens.iter().enumerate() {
            if let Some(byte) = classes.first_byte[index] {
                if !self.first_byte_allowed(byte) {
                    continue;
                }
            }
            let mut probe = self.clone();
            if probe.push_str(token) {
                allowed.push(index as u32);
            }
        }
        allowed
    }

    /// [`Pda::mask`] over a pre-classified vocabulary.
    pub fn mask_classified(&self, classes: &TokenClasses, eos: Option<&str>) -> Vec<u32> {
        let mut allowed = self.candidates_classified(classes);
        if self.finish().is_some() {
            for (index, token) in classes.tokens.iter().enumerate() {
                if Some(token.as_str()) == eos {
                    allowed.push(index as u32);
                }
            }
        }
        allowed.sort_unstable();
        allowed.dedup();
        allowed
    }

    /// Indices of tokens that keep the document alive.
    pub fn candidates(&self, vocabulary: &[String]) -> Vec<u32> {
        let mut allowed = Vec::new();
        for (index, token) in vocabulary.iter().enumerate() {
            let mut probe = self.clone();
            if probe.push_str(token) {
                allowed.push(index as u32);
            }
        }
        allowed
    }

    /// Candidates plus the end-of-sequence token, which is only legal on a finished document.
    pub fn mask(&self, vocabulary: &[String], eos: Option<&str>) -> Vec<u32> {
        let mut allowed = self.candidates(vocabulary);
        let finished = self.finish().is_some();
        if finished {
            for (index, token) in vocabulary.iter().enumerate() {
                if Some(token.as_str()) == eos {
                    allowed.push(index as u32);
                }
            }
        }
        allowed.sort_unstable();
        allowed.dedup();
        allowed
    }
}

/// Per-vocabulary facts computed once and reused on every decoding step.
///
/// `first_byte` is `None` for the empty token (a no-op that never changes state) and for
/// tokens whose first character is not ASCII, which must always be replayed.
#[derive(Clone, Debug)]
pub struct TokenClasses {
    tokens: Vec<String>,
    first_byte: Vec<Option<u8>>,
}

impl TokenClasses {
    pub fn new(vocabulary: &[String]) -> TokenClasses {
        TokenClasses {
            tokens: vocabulary.to_vec(),
            first_byte: vocabulary
                .iter()
                .map(|token| match token.as_bytes().first() {
                    None => None,
                    Some(byte) if byte.is_ascii() => Some(*byte),
                    Some(_) => None,
                })
                .collect(),
        }
    }

    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Tokens that the partition keeps dynamic (non-ASCII or structurally relevant).
    pub fn dynamic_len(&self, pda: &Pda) -> usize {
        self.first_byte
            .iter()
            .filter(|byte| match byte {
                None => true,
                Some(byte) => pda.first_byte_allowed(*byte),
            })
            .count()
    }
}

/// Does the literal already spell an exponent?
fn has_exponent(text: &str) -> bool {
    text.as_bytes().iter().any(|b| *b == b'e' || *b == b'E')
}

/// Is the exponent already explicitly negative (so more digits lower the magnitude)?
fn exponent_can_shrink(text: &str) -> bool {
    match text.rfind(['e', 'E']) {
        Some(index) => text.as_bytes().get(index + 1) == Some(&b'-'),
        None => false,
    }
}

/// A complete number must be representable as a finite IEEE-754 value; ``json::parse``
/// refuses ``NaN``/``inf`` and the masker must not admit a token the parser rejects.
fn number_is_finite(text: &str) -> bool {
    text.parse::<f64>().map(|value| value.is_finite()).unwrap_or(false)
}

/// Is `text` a prefix of a JSON number (`complete = false`) or a whole one (`complete = true`)?
pub fn number_prefix_ok(text: &str, complete: bool) -> bool {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let mut at = 0usize;
    if bytes[at] == b'-' {
        at += 1;
        if at == bytes.len() {
            return !complete;
        }
    }
    match bytes[at] {
        b'0' => at += 1,
        b'1'..=b'9' => {
            while at < bytes.len() && bytes[at].is_ascii_digit() {
                at += 1;
            }
        }
        _ => return false,
    }
    if at == bytes.len() {
        return true;
    }
    if bytes[at] == b'.' {
        at += 1;
        let start = at;
        while at < bytes.len() && bytes[at].is_ascii_digit() {
            at += 1;
        }
        if at == bytes.len() {
            return !complete || at > start;
        }
        if at == start {
            return false;
        }
    }
    if at < bytes.len() && (bytes[at] == b'e' || bytes[at] == b'E') {
        at += 1;
        if at < bytes.len() && (bytes[at] == b'+' || bytes[at] == b'-') {
            at += 1;
        }
        let start = at;
        while at < bytes.len() && bytes[at].is_ascii_digit() {
            at += 1;
        }
        if at == bytes.len() {
            return !complete || at > start;
        }
        if at == start {
            return false;
        }
    }
    at == bytes.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::prefix_status;
    use crate::json::Prefix;

    fn feed(text: &str) -> Pda {
        let mut pda = Pda::new();
        pda.push_str(text);
        pda
    }

    #[test]
    fn accepts_complete_documents_only() {
        for text in [
            "1",
            "-0.5",
            "1e3",
            "\"x\"",
            "true",
            "null",
            "[]",
            "{}",
            "{\"a\": [1, 2, {\"b\": null}], \"c\": false}",
            "  { \"a\" : 1 }  ",
        ] {
            let pda = feed(text);
            assert!(pda.accepts(), "should accept {:?}", text);
        }
        for text in ["", "{", "{\"a\"", "{\"a\":1,}", "[1,", "\"abc", "1e", "-", "truex"] {
            let pda = feed(text);
            assert!(!pda.accepts(), "should not accept {:?}", text);
        }
    }

    #[test]
    fn events_track_structure() {
        let pda = feed("{\"a\": [1]}");
        assert_eq!(pda.last_event(), Some(Event::Value));
        assert!(pda.events().contains(&Event::Key));
        // One close for the array, one for the object.
        assert_eq!(
            pda.events().iter().filter(|event| **event == Event::Close).count(),
            2
        );
        assert_eq!(pda.depth(), 0);
    }

    #[test]
    fn prefix_verdicts_match_the_json_recognizer() {
        let samples = [
            "",
            " ",
            "{",
            "{}",
            "{\"a\"",
            "{\"a\":",
            "{\"a\": ",
            "{\"a\": 1",
            "{\"a\": 1}",
            "{\"a\" 1}",
            "[1,]",
            "[1",
            "nul",
            "null",
            "nullx",
            "\"a\\",
            "\"a\\u00",
            "0.",
            "01",
            "-",
            "1e",
            "1e+",
            "1e+2",
            "abc",
        ];
        for sample in samples {
            let pda = feed(sample);
            let expected = prefix_status(sample);
            let actual = if pda.accepts() {
                Prefix::Complete
            } else if pda.is_alive() {
                Prefix::Incomplete
            } else {
                Prefix::Invalid
            };
            assert_eq!(actual, expected, "diverged on {:?}", sample);
        }
    }

    #[test]
    fn mask_admits_only_parseable_tokens() {
        let vocabulary: Vec<String> = [
            "{\"a\": ",
            "}",
            "\"a\"",
            ": ",
            "1",
            "]",
            "null",
            "<stop>",
        ]
        .iter()
        .map(|token| token.to_string())
        .collect();
        let empty = Pda::new();
        let allowed = empty.mask(&vocabulary, Some("<stop>"));
        assert!(allowed.contains(&0)); // a key opens the document
        assert!(allowed.contains(&6)); // so does a literal: null is a document
        assert!(!allowed.contains(&5)); // a bare ] cannot start anything
        assert!(!allowed.contains(&7)); // and the document is not finished yet

        let open = feed("{\"a\": ");
        let allowed = open.mask(&vocabulary, Some("<stop>"));
        assert!(allowed.contains(&4));
        assert!(allowed.contains(&6));
        assert!(!allowed.contains(&1));

        let done = feed("{\"a\": 1}");
        assert_eq!(done.mask(&vocabulary, Some("<stop>")), vec![7]);
    }

    #[test]
    fn number_prefixes_follow_rfc_8259() {
        for text in ["0", "-0", "1", "-1.5", "1e3", "1E+3", "0.25"] {
            assert!(number_prefix_ok(text, true), "{} should be a number", text);
        }
        for text in ["", "-", "01", "1.", ".5", "1e", "1e+", "+1", "1x", "00"] {
            assert!(!number_prefix_ok(text, true), "{} is not a number", text);
        }
        for text in ["-", "1.", "1e", "1e+", "0."] {
            assert!(number_prefix_ok(text, false), "{} is a legal prefix", text);
        }
        // An empty string is a prefix of *a value*, not of a number.
        assert!(!number_prefix_ok("", false));
    }

    #[test]
    fn overflowing_numbers_die_like_the_strict_parser() {
        // ``json::parse`` rejects non-finite values; the automaton must agree, otherwise a
        // mask admits a token the decoder's own verifier then refuses.
        for text in ["72E934", "[1e400]", "1e400", "[90e9376", "-9e999"] {
            assert!(!feed(text).is_alive(), "{text} must be dead");
        }
        // A long mantissa is still rescuable by a negative exponent.
        let mut rescued = "9".repeat(400);
        rescued.push_str("e-400");
        assert!(feed(&rescued).is_alive(), "long mantissa plus e-400 is finite");
        // ... but a finished overflowing mantissa is not.
        assert!(!feed(&"9".repeat(400)).accepts());
    }

    #[test]
    fn classification_matches_the_full_replay() {
        let vocabulary: Vec<String> = [
            "{\"a\": ",
            "}",
            "]",
            "\"a\"",
            ": ",
            "1",
            "-2.5e3",
            "null",
            "tru",
            " \n",
            "é",
            "<stop>",
            "\"",
            "\\u00e9",
            "",
        ]
        .iter()
        .map(|token| token.to_string())
        .collect();
        let classes = TokenClasses::new(&vocabulary);
        for prefix in ["", "{", "{\"a\": ", "[1", "null", "\"x\"", "{\"a\": 1}"] {
            let pda = feed(prefix);
            assert_eq!(
                pda.mask(&vocabulary, Some("<stop>")),
                pda.mask_classified(&classes, Some("<stop>")),
                "partition diverged on {:?}",
                prefix
            );
        }
        // The empty token is a no-op and must never be filtered out.
        assert_eq!(classes.first_byte[vocabulary.len() - 1], None);
        assert!(Pda::new().candidates_classified(&classes).contains(&14));

        // A dead machine offers nothing; structure narrows the dynamic subset considerably.
        let dead = feed("{\"a\" 1");
        assert!(!dead.first_byte_allowed(b'"'));
        // Only the empty token survives on a dead machine: it changes no state, so replaying
        // it is still a no-op rather than a resurrection.
        assert!(dead
            .candidates_classified(&classes)
            .iter()
            .all(|index| *index as usize == vocabulary.len() - 1));
        let structural = Pda::new();
        assert!(classes.dynamic_len(&structural) < classes.len());
    }

    #[test]
    fn literals_are_consumed_incrementally() {
        let mut pda = Pda::new();
        assert!(pda.push_str("tru"));
        assert!(pda.is_alive() && !pda.accepts());
        assert!(pda.push_str("e"));
        assert!(pda.accepts());
        let mut broken = Pda::new();
        assert!(!broken.push_str("trux"));
        assert!(!broken.is_alive());
    }
}
