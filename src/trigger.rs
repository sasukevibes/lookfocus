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
//! The trigger takes one sample at a time with its time, so tests drive it with
//! made-up times.

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
}
