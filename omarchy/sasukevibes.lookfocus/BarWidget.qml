pragma ComponentBehavior: Bound

import QtQuick
import Quickshell
import Quickshell.Io
import qs.Commons
import qs.Ui

// lookfocus in the Omarchy bar.
//
// Left click pauses or resumes tracking (starting the service if it is not
// running). Right click opens a dropdown with tracking, adaptive centroids,
// recalibration and the service. Everything goes through the `lookfocus`
// CLI, the same one the keybind uses, so the bar never talks to the daemon
// directly.
Panel {
  id: root
  moduleName: "sasukevibes.lookfocus"
  ipcTarget: "sasukevibes.lookfocus"
  implicitWidth: button.implicitWidth
  implicitHeight: Style.bar.sizeHorizontal

  // `lookfocus status --json`, or a placeholder until the first answer.
  property var status: ({ running: false, state: "unknown", adaptive: false, drift: [] })
  property bool cliMissing: false

  readonly property string stateName: String(status.state || "unknown")
  readonly property bool running: !!status.running
  readonly property bool tracking: stateName === "tracking" || stateName === "away"
  readonly property bool problem: stateName === "layout_changed" || stateName === "camera_unavailable"
    || stateName === "not_calibrated" || cliMissing

  // Runs the CLI with ~/.local/bin on PATH, since the shell's own PATH may
  // not include it.
  function cli(args) {
    return ["sh", "-c", "PATH=\"$HOME/.local/bin:$PATH\" exec lookfocus \"$@\"", "lookfocus"].concat(args)
  }

  function refresh() {
    if (!poll.running) poll.running = true
  }

  function run(args) {
    action.command = root.cli(args)
    action.running = true
  }

  function icon() {
    if (problem) return "󰀦"
    if (stateName === "paused" || stateName === "stopped") return "󰈉"
    return "󰈈"
  }

  function label() {
    if (stateName === "tracking" && status.monitor) return icon() + " " + status.monitor
    return icon()
  }

  function stateText() {
    switch (stateName) {
    case "tracking": return "Tracking · focused on " + (status.monitor || "?")
      + (status.zone ? " · facing " + status.zone : "")
    case "away": return "Away · camera released until you are back"
    case "paused": return "Paused · camera released"
    case "stopped": return "Not running"
    case "not_calibrated": return "Not calibrated yet"
    case "layout_changed": return "Monitor layout changed · recalibrate"
    case "camera_unavailable": return "Camera unavailable"
    default: return cliMissing ? "lookfocus is not installed" : "Checking…"
    }
  }

  function learnedText() {
    var parts = []
    var d = status.drift || []
    for (var i = 0; i < d.length; i++) {
      if (d[i][1] > 0.05) parts.push(d[i][0] + " " + Number(d[i][1]).toFixed(1) + "°")
    }
    return parts.length ? "Learned: " + parts.join(", ") : "Learns where you look from mouse use"
  }

  function tip() {
    var t = "lookfocus · " + stateText()
    if (status.reason && !tracking) t += "\n" + status.reason
    return t + "\nClick to " + (tracking ? "pause" : "resume") + " · right-click for options"
  }

  Process {
    id: poll
    command: root.cli(["status", "--json"])
    stdout: StdioCollector {
      onStreamFinished: {
        try {
          root.status = JSON.parse(text)
          root.cliMissing = false
        } catch (e) {}
      }
    }
    onExited: function(code) {
      if (code === 127) root.cliMissing = true
    }
  }

  Process {
    id: action
    onExited: root.refresh()
  }

  Process {
    id: detached
  }

  Timer {
    interval: root.opened ? 1000 : 3000
    running: true; repeat: true; triggeredOnStart: true
    onTriggered: root.refresh()
  }

  WidgetButton {
    id: button
    anchors.fill: parent
    bar: root.bar
    text: root.label()
    dimmed: !root.tracking && !root.problem
    fontSize: Style.font.caption
    tooltipText: root.tip()
    onPressed: function(mouseButton) {
      if (mouseButton === Qt.RightButton) root.toggle()
      else root.run(["toggle"])
    }
  }

  KeyboardPanel {
    id: popup
    anchorItem: button
    owner: root
    bar: root.bar
    open: root.opened
    contentWidth: popup.fittedContentWidth(Style.space(340))
    contentHeight: popup.fittedContentHeight(content.implicitHeight)

    Column {
      id: content
      width: parent.width
      spacing: Style.space(8)

      Text {
        text: "lookfocus"
        color: Color.foreground
        font.family: Style.font.family
        font.pixelSize: Style.font.title
        font.bold: true
      }

      Text {
        width: parent.width
        text: root.stateText()
        wrapMode: Text.WordWrap
        color: Qt.darker(Color.foreground, 1.2)
        font.family: Style.font.family
        font.pixelSize: Style.font.caption
      }

      Toggle {
        width: parent.width
        label: "Tracking"
        description: root.tracking ? "Focus follows your head · SUPER+ALT+E" : "Off · camera released"
        checked: root.tracking
        onClicked: root.run([root.tracking ? "pause" : "resume"])
      }

      Toggle {
        width: parent.width
        label: "Adaptive centroids"
        description: root.learnedText()
        checked: !!root.status.adaptive
        onClicked: root.run(["adaptive", "toggle"])
      }

      PanelSeparator {}

      Row {
        spacing: Style.space(8)

        Button {
          iconText: "󰑓"
          text: "Recalibrate"
          tooltipText: "Opens a terminal and walks through calibration"
          foreground: Color.foreground
          onClicked: {
            root.close()
            // Resolve the full path here: the terminal's shell may not
            // have ~/.local/bin on its PATH.
            detached.command = ["sh", "-c",
              "PATH=\"$HOME/.local/bin:$PATH\"; bin=$(command -v lookfocus) || exit 1; "
              + "exec omarchy-launch-floating-terminal-with-presentation \"$bin\" recalibrate"]
            detached.running = true
          }
        }

        Button {
          iconText: root.running ? "󰓛" : "󰐊"
          text: root.running ? "Stop service" : "Start service"
          foreground: Color.foreground
          onClicked: {
            action.command = ["systemctl", "--user", root.running ? "stop" : "start", "lookfocus.service"]
            action.running = true
          }
        }
      }

      Text {
        width: parent.width
        visible: !!root.status.reason && !root.tracking
        text: root.status.reason || ""
        wrapMode: Text.WordWrap
        color: Qt.darker(Color.foreground, 1.5)
        font.family: Style.font.family
        font.pixelSize: Style.font.caption
      }

      Text {
        width: parent.width
        text: "lookfocus never stores or sends camera frames, and never reads your keyboard."
        wrapMode: Text.WordWrap
        color: Qt.darker(Color.foreground, 1.6)
        font.family: Style.font.family
        font.pixelSize: Style.font.caption
      }
    }
  }
}
