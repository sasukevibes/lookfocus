//! Internal events.
//!
//! Everything the daemon does that another program might care about goes
//! through here as an `Event`. Today the subscribers are the log and tests. A
//! later version will stream these as JSON lines on a Unix socket, so other
//! tools can react to where you are looking. That only needs a new
//! subscriber, not changes to the daemon.

use std::sync::mpsc::{Receiver, Sender, channel};

use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    /// Tracking started with these calibrated monitors.
    Started {
        monitors: Vec<String>,
    },
    FaceFound,
    FaceLost,
    /// The monitor your head points at changed. This can change without a
    /// switch, for example during a quick glance.
    Zone {
        monitor: String,
    },
    /// lookfocus moved focus.
    Switched {
        from: Option<String>,
        to: String,
    },
    /// Focus moved for another reason (mouse, keybind).
    FocusChanged {
        monitor: String,
    },
    /// Switching stopped, with a reason such as a monitor layout change.
    Paused {
        reason: String,
    },
    Resumed,
    /// No face for a while, so the camera was released.
    Away,
    /// A face is back after being away.
    Back,
    /// The camera could not be opened (busy, unplugged, no permission).
    CameraUnavailable {
        reason: String,
    },
    AdaptiveChanged {
        enabled: bool,
    },
}

#[derive(Default)]
pub struct EventBus {
    subscribers: Vec<Sender<Event>>,
}

impl EventBus {
    pub fn subscribe(&mut self) -> Receiver<Event> {
        let (tx, rx) = channel();
        self.subscribers.push(tx);
        rx
    }

    /// Adds a subscriber made elsewhere, such as a `watch` client.
    pub fn add(&mut self, tx: Sender<Event>) {
        self.subscribers.push(tx);
    }

    /// Sends to every subscriber, dropping the ones that have gone away.
    pub fn emit(&mut self, event: Event) {
        log::debug!("event: {event:?}");
        self.subscribers.retain(|s| s.send(event.clone()).is_ok());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delivers_to_all_and_drops_closed_subscribers() {
        let mut bus = EventBus::default();
        let a = bus.subscribe();
        let b = bus.subscribe();
        bus.emit(Event::FaceFound);
        drop(b);
        bus.emit(Event::Zone { monitor: "DP-2".into() });
        assert_eq!(a.try_iter().collect::<Vec<_>>(), vec![Event::FaceFound, Event::Zone { monitor: "DP-2".into() }]);
        assert_eq!(bus.subscribers.len(), 1);
    }

    #[test]
    fn serializes_as_tagged_json() {
        let e = Event::Switched { from: Some("DP-2".into()), to: "eDP-2".into() };
        assert_eq!(serde_json::to_string(&e).unwrap(), r#"{"event":"switched","from":"DP-2","to":"eDP-2"}"#);
    }
}
