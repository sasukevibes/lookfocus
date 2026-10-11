//! When a hand gesture counts as a deliberate command.
//!
//! The camera sees a hand many times a second, and people wave their hands
//! about without meaning anything by it. A gesture fires only when:
//!
//! - it has been held for `hold` without a break,
//! - a face is in view while it is held (unless that is switched off),
//! - it is armed: after a gesture fires, it must be gone from the picture for
//!   half a second before it can fire again, so holding a palm up fires once
//!   and not over and over, and
//! - the last firing was at least `cooldown` ago. A gesture held through the
//!   cooldown fires when it ends.
//!
//! That is `Trigger`, for toggle mode: a held gesture fires once. Hold mode
//! ("hold to talk") uses `HoldTrigger` instead. A hold starts under the same
//! rules (held for `hold`, a face in view, `cooldown` since the last start) and
//! then stays on for as long as the gesture is held. It ends once the gesture
//! has been gone for `release`, so a frame or two without the hand does not end
//! it. The face is needed only to start: while a hold is on, losing the face
//! changes nothing.
//!
//! The triggers take one sample at a time with its time, so tests drive them
//! with made-up times.

use std::time::{Duration, Instant};

use crate::gesture::Gesture;

/// How long a fired gesture must be gone before it can fire again.
pub const REARM_AFTER: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TriggerParams {
    /// How long a gesture must be held before it fires.
    pub hold: Duration,
    /// The least time between two firings.
    pub cooldown: Duration,
    /// Count a gesture only while a face is in view.
    pub require_face: bool,
}

/// A gesture that fired and has not been gone long enough to fire again.
struct Spent {
    gesture: Gesture,
    /// When it first went missing, if it has.
    gone_since: Option<Instant>,
}

pub struct Trigger {
    params: TriggerParams,
    /// The gesture being held and when it started.
    held: Option<(Gesture, Instant)>,
    spent: Option<Spent>,
    last_fired: Option<Instant>,
}

impl Trigger {
    pub fn new(params: TriggerParams) -> Self {
        Self { params, held: None, spent: None, last_fired: None }
    }

    /// Feeds one sample: the gesture seen in it (None for no hand, or a hand
    /// that shows no known gesture) and whether a face was in view. Returns the
    /// gesture if it fires on this sample.
    pub fn update(&mut self, now: Instant, seen: Option<Gesture>, face: bool) -> Option<Gesture> {
        // Re-arming goes by what the camera sees, with a face or without.
        if let Some(spent) = &mut self.spent {
            if seen == Some(spent.gesture) {
                spent.gone_since = None;
            } else if now.duration_since(*spent.gone_since.get_or_insert(now)) >= REARM_AFTER {
                self.spent = None;
            }
        }

        let counted = seen.filter(|_| face || !self.params.require_face);
        match (counted, self.held) {
            (Some(g), Some((held, _))) if g == held => {}
            (Some(g), _) => self.held = Some((g, now)),
            (None, _) => self.held = None,
        }

        let (gesture, since) = self.held?;
        let held_long_enough = now.duration_since(since) >= self.params.hold;
        let armed = self.spent.as_ref().is_none_or(|s| s.gesture != gesture);
        let cooled = self.last_fired.is_none_or(|t| now.duration_since(t) >= self.params.cooldown);
        if !(held_long_enough && armed && cooled) {
            return None;
        }
        self.last_fired = Some(now);
        self.spent = Some(Spent { gesture, gone_since: None });
        Some(gesture)
    }

    /// Forgets what is being held and what has fired, for example when the
    /// camera closes. The cooldown stays, so closing the camera does not
    /// skip it.
    pub fn reset(&mut self) {
        self.held = None;
        self.spent = None;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HoldParams {
    /// How long a gesture must be held before the hold starts.
    pub hold: Duration,
    /// How long the gesture must be gone before the hold ends.
    pub release: Duration,
    /// The least time between two starts.
    pub cooldown: Duration,
    /// Start a hold only while a face is in view.
    pub require_face: bool,
}

/// What a sample did to a hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoldEvent {
    Started(Gesture),
    Ended(Gesture),
}

/// A hold that has started and not ended.
struct Active {
    gesture: Gesture,
    /// When the gesture went missing, if it has.
    gone_since: Option<Instant>,
}

pub struct HoldTrigger {
    params: HoldParams,
    /// The gesture being held toward a start, and when that began.
    held: Option<(Gesture, Instant)>,
    active: Option<Active>,
    last_started: Option<Instant>,
}

impl HoldTrigger {
    pub fn new(params: HoldParams) -> Self {
        Self { params, held: None, active: None, last_started: None }
    }

    /// Feeds one sample, as `Trigger::update` does. Returns whether a hold
    /// started or ended on it.
    pub fn update(&mut self, now: Instant, seen: Option<Gesture>, face: bool) -> Option<HoldEvent> {
        if let Some(active) = &mut self.active {
            if seen == Some(active.gesture) {
                active.gone_since = None;
                return None;
            }
            if now.duration_since(*active.gone_since.get_or_insert(now)) < self.params.release {
                return None;
            }
            return self.end().map(HoldEvent::Ended);
        }

        let counted = seen.filter(|_| face || !self.params.require_face);
        match (counted, self.held) {
            (Some(g), Some((held, _))) if g == held => {}
            (Some(g), _) => self.held = Some((g, now)),
            (None, _) => self.held = None,
        }

        let (gesture, since) = self.held?;
        let held_long_enough = now.duration_since(since) >= self.params.hold;
        let cooled = self.last_started.is_none_or(|t| now.duration_since(t) >= self.params.cooldown);
        if !(held_long_enough && cooled) {
            return None;
        }
        self.held = None;
        self.last_started = Some(now);
        self.active = Some(Active { gesture, gone_since: None });
        Some(HoldEvent::Started(gesture))
    }

    /// The gesture of the hold that is on, if one is.
    pub fn active(&self) -> Option<Gesture> {
        self.active.as_ref().map(|a| a.gesture)
    }

    /// Ends the hold that is on now, whatever the camera sees, and forgets any
    /// gesture being held toward a start. Returns the gesture of the hold it
    /// ended, so its stop command can run. The cooldown stays.
    pub fn end(&mut self) -> Option<Gesture> {
        self.held = None;
        self.active.take().map(|a| a.gesture)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: Duration = Duration::from_millis(66);

    fn params() -> TriggerParams {
        TriggerParams { hold: Duration::from_millis(400), cooldown: Duration::from_millis(1500), require_face: true }
    }

    /// A trigger with a clock that moves one frame per sample.
    struct Run {
        t: Instant,
        trigger: Trigger,
    }

    impl Run {
        fn new(params: TriggerParams) -> Self {
            Self { t: Instant::now(), trigger: Trigger::new(params) }
        }

        /// Feeds `frames` identical samples and returns how many times it fired.
        fn feed(&mut self, seen: Option<Gesture>, face: bool, frames: usize) -> usize {
            (0..frames)
                .filter(|_| {
                    self.t += FRAME;
                    self.trigger.update(self.t, seen, face).is_some()
                })
                .count()
        }

        fn palm(&mut self, frames: usize) -> usize {
            self.feed(Some(Gesture::OpenPalm), true, frames)
        }

        fn nothing(&mut self, frames: usize) -> usize {
            self.feed(None, true, frames)
        }
    }

    #[test]
    fn a_held_palm_fires_once() {
        let mut r = Run::new(params());
        assert_eq!(r.palm(60), 1);
    }

    #[test]
    fn it_fires_after_the_hold_time_and_not_before() {
        let mut r = Run::new(params());
        // The first sample starts the hold. The 8th is 462 ms later, and the
        // 7th only 396 ms.
        assert_eq!(r.palm(7), 0);
        assert_eq!(r.palm(1), 1);
    }

    #[test]
    fn a_short_palm_does_not_fire() {
        let mut r = Run::new(params());
        assert_eq!(r.palm(5), 0);
        assert_eq!(r.nothing(30), 0);
    }

    #[test]
    fn a_broken_hold_starts_over() {
        let mut r = Run::new(params());
        assert_eq!(r.palm(5), 0);
        assert_eq!(r.nothing(1), 0);
        assert_eq!(r.palm(5), 0);
    }

    #[test]
    fn it_fires_again_after_the_gesture_was_gone_long_enough() {
        let mut r = Run::new(params());
        assert_eq!(r.palm(20), 1);
        assert_eq!(r.nothing(12), 0); // 790 ms
        assert_eq!(r.palm(20), 1);
    }

    #[test]
    fn a_short_break_does_not_re_arm() {
        let mut r = Run::new(params());
        assert_eq!(r.palm(20), 1);
        assert_eq!(r.nothing(5), 0); // 330 ms
        assert_eq!(r.palm(60), 0);
    }

    #[test]
    fn another_gesture_in_between_still_counts_as_gone() {
        let mut r = Run::new(params());
        assert_eq!(r.palm(20), 1);
        // A hand that shows no known gesture is not the palm.
        assert_eq!(r.nothing(12), 0);
        assert_eq!(r.palm(20), 1);
    }

    #[test]
    fn the_cooldown_holds_back_a_second_firing() {
        let mut r = Run::new(TriggerParams { cooldown: Duration::from_secs(5), ..params() });
        assert_eq!(r.palm(10), 1);
        assert_eq!(r.nothing(12), 0);
        // Re-armed and held long enough, but only about 2 s since the first.
        assert_eq!(r.palm(20), 0);
    }

    #[test]
    fn a_gesture_held_through_the_cooldown_fires_when_it_ends() {
        let mut r = Run::new(TriggerParams { cooldown: Duration::from_secs(3), ..params() });
        assert_eq!(r.palm(10), 1);
        assert_eq!(r.nothing(12), 0);
        // Held for 5 s: the cooldown ends part way through.
        assert_eq!(r.palm(75), 1);
    }

    #[test]
    fn without_a_face_nothing_counts() {
        let mut r = Run::new(params());
        assert_eq!(r.feed(Some(Gesture::OpenPalm), false, 60), 0);
        // The face coming back starts the hold from there.
        assert_eq!(r.palm(5), 0);
        assert_eq!(r.palm(5), 1);
    }

    #[test]
    fn losing_the_face_mid_hold_starts_over() {
        let mut r = Run::new(params());
        assert_eq!(r.palm(5), 0);
        assert_eq!(r.feed(Some(Gesture::OpenPalm), false, 1), 0);
        assert_eq!(r.palm(5), 0);
    }

    #[test]
    fn a_face_is_not_needed_when_not_required() {
        let mut r = Run::new(TriggerParams { require_face: false, ..params() });
        assert_eq!(r.feed(Some(Gesture::OpenPalm), false, 60), 1);
    }

    #[test]
    fn a_palm_held_through_a_lost_face_does_not_re_arm() {
        let mut r = Run::new(params());
        assert_eq!(r.palm(20), 1);
        // The face drops out for a second while the palm stays up.
        assert_eq!(r.feed(Some(Gesture::OpenPalm), false, 15), 0);
        assert_eq!(r.palm(60), 0);
    }

    #[test]
    fn reset_forgets_the_hold_and_the_firing_but_keeps_the_cooldown() {
        let quick = TriggerParams { hold: Duration::from_millis(100), ..params() };
        let mut r = Run::new(quick);
        assert_eq!(r.palm(10), 1);
        // Without a reset the palm would stay spent. After one it is armed
        // again at once.
        r.trigger.reset();
        r.t += Duration::from_secs(2);
        assert_eq!(r.palm(10), 1);

        // The cooldown survives the reset.
        let mut r = Run::new(quick);
        assert_eq!(r.palm(10), 1);
        r.trigger.reset();
        assert_eq!(r.palm(10), 0, "still cooling down");
    }

    // ------------------------------------------------------------ hold

    fn hold_params() -> HoldParams {
        HoldParams {
            hold: Duration::from_millis(400),
            release: Duration::from_millis(500),
            cooldown: Duration::from_millis(1500),
            require_face: true,
        }
    }

    /// A hold trigger with a clock that moves one frame per sample.
    struct HoldRun {
        t: Instant,
        trigger: HoldTrigger,
    }

    impl HoldRun {
        fn new(params: HoldParams) -> Self {
            Self { t: Instant::now(), trigger: HoldTrigger::new(params) }
        }

        /// Feeds `frames` identical samples and returns what happened.
        fn feed(&mut self, seen: Option<Gesture>, face: bool, frames: usize) -> Vec<HoldEvent> {
            (0..frames)
                .filter_map(|_| {
                    self.t += FRAME;
                    self.trigger.update(self.t, seen, face)
                })
                .collect()
        }

        fn palm(&mut self, frames: usize) -> Vec<HoldEvent> {
            self.feed(Some(Gesture::OpenPalm), true, frames)
        }

        fn nothing(&mut self, frames: usize) -> Vec<HoldEvent> {
            self.feed(None, true, frames)
        }
    }

    const STARTED: HoldEvent = HoldEvent::Started(Gesture::OpenPalm);
    const ENDED: HoldEvent = HoldEvent::Ended(Gesture::OpenPalm);

    #[test]
    fn a_hold_starts_after_the_hold_time_and_lasts_while_held() {
        let mut r = HoldRun::new(hold_params());
        assert_eq!(r.palm(7), vec![]);
        assert_eq!(r.palm(1), vec![STARTED]);
        assert_eq!(r.trigger.active(), Some(Gesture::OpenPalm));
        // Held for a long while: nothing more happens.
        assert_eq!(r.palm(300), vec![]);
        assert_eq!(r.trigger.active(), Some(Gesture::OpenPalm));
    }

    #[test]
    fn a_hold_ends_once_the_gesture_has_been_gone_for_the_release_time() {
        let mut r = HoldRun::new(hold_params());
        assert_eq!(r.palm(20), vec![STARTED]);
        // The first frame without it starts the release; 7 frames later is 462 ms.
        assert_eq!(r.nothing(8), vec![]);
        assert_eq!(r.nothing(1), vec![ENDED]);
        assert_eq!(r.trigger.active(), None);
        assert_eq!(r.nothing(30), vec![]);
    }

    #[test]
    fn a_short_palm_does_not_start_a_hold() {
        let mut r = HoldRun::new(hold_params());
        assert_eq!(r.palm(5), vec![]);
        assert_eq!(r.nothing(30), vec![]);
        assert_eq!(r.trigger.active(), None);
    }

    #[test]
    fn a_brief_loss_of_the_hand_does_not_end_a_hold() {
        let mut r = HoldRun::new(hold_params());
        assert_eq!(r.palm(10), vec![STARTED]);
        for _ in 0..5 {
            assert_eq!(r.nothing(5), vec![]); // 330 ms
            assert_eq!(r.palm(10), vec![]);
        }
        // A hand showing something else is the same as no hand.
        assert_eq!(r.nothing(6), vec![]);
        assert_eq!(r.trigger.active(), Some(Gesture::OpenPalm));
    }

    #[test]
    fn the_face_is_needed_to_start_and_not_to_keep_a_hold() {
        let mut r = HoldRun::new(hold_params());
        assert_eq!(r.feed(Some(Gesture::OpenPalm), false, 60), vec![]);
        assert_eq!(r.palm(8), vec![STARTED]);
        // The palm covers the face for a long time: the hold stays on.
        assert_eq!(r.feed(Some(Gesture::OpenPalm), false, 200), vec![]);
        // And the release works without a face too.
        assert_eq!(r.feed(None, false, 9), vec![ENDED]);
    }

    #[test]
    fn a_hold_can_start_without_a_face_when_not_required() {
        let mut r = HoldRun::new(HoldParams { require_face: false, ..hold_params() });
        assert_eq!(r.feed(Some(Gesture::OpenPalm), false, 8), vec![STARTED]);
    }

    #[test]
    fn the_cooldown_spaces_starts() {
        let mut r = HoldRun::new(HoldParams { cooldown: Duration::from_secs(3), ..hold_params() });
        assert_eq!(r.palm(8), vec![STARTED]);
        assert_eq!(r.nothing(9), vec![ENDED]); // about 1.1 s after the start
        // Held long enough again, but the cooldown is not over until about
        // 3 s after the start: it starts then, while still held.
        let events = r.palm(40);
        assert_eq!(events, vec![STARTED]);
        let mut r = HoldRun::new(HoldParams { cooldown: Duration::from_secs(3), ..hold_params() });
        assert_eq!(r.palm(8), vec![STARTED]);
        assert_eq!(r.nothing(9), vec![ENDED]);
        assert_eq!(r.palm(20), vec![], "1.3 s more is still inside the cooldown");
    }

    #[test]
    fn a_new_hold_needs_the_full_hold_time_again() {
        let mut r = HoldRun::new(HoldParams { cooldown: Duration::ZERO, ..hold_params() });
        assert_eq!(r.palm(8), vec![STARTED]);
        assert_eq!(r.nothing(9), vec![ENDED]);
        assert_eq!(r.palm(7), vec![]);
        assert_eq!(r.palm(1), vec![STARTED]);
    }

    #[test]
    fn end_stops_a_hold_at_once_and_says_which() {
        let mut r = HoldRun::new(HoldParams { cooldown: Duration::ZERO, ..hold_params() });
        assert_eq!(r.trigger.end(), None);
        assert_eq!(r.palm(8), vec![STARTED]);
        assert_eq!(r.trigger.end(), Some(Gesture::OpenPalm));
        assert_eq!(r.trigger.active(), None);
        assert_eq!(r.trigger.end(), None, "only once");
        // A palm still up after that has to be held again to start again.
        assert_eq!(r.palm(7), vec![]);
        assert_eq!(r.palm(1), vec![STARTED]);
    }

    #[test]
    fn end_keeps_the_cooldown() {
        let mut r = HoldRun::new(hold_params());
        assert_eq!(r.palm(8), vec![STARTED]);
        r.trigger.end();
        assert_eq!(r.palm(10), vec![], "still cooling down");
    }
}
