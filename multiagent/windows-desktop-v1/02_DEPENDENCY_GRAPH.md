# Dependency Graph

```mermaid
graph TD
  T01["T01: Core contracts and worker"]
  T02["T02: PySide6 desktop"]
  T03["T03: Launcher and packaging"]
  T04["T04: Synthetic public assets"]
  T05["T05: Algorithm hardening"]
  T06["T06: Cross-review"]
  TF["T_FINAL: Integration"]

  T01 --> T02
  T04 --> T02
  T01 --> T05
  T02 --> T06
  T03 --> T06
  T04 --> T06
  T05 --> T06
  T01 --> TF
  T02 --> TF
  T03 --> TF
  T04 --> TF
  T05 --> TF
  T06 --> TF
```

## Parallel Launch Groups

| Group | Can Start | Must Wait For |
| --- | --- | --- |
| A | T01, T03, T04, T05 | None; T05 records contract requests for T01/T_FINAL |
| B | T02 | T01 and T04 handoffs |
| C | T06 | T02–T05 handoffs and smoke results |
| D | T_FINAL | All Txx handoffs and smoke results |
