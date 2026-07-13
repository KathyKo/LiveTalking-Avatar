# LiveTalking API Reference

Base path: `http://<host>:<listenport>`

All endpoints return a unified response format:

```json
{ "code": 0, "msg": "ok", "data": {} }
```

A `code` of 0 indicates success; a non-zero value indicates an error.

---

## 1. WebRTC Offer

Exchange SDP to establish a WebRTC connection.

```
POST /offer
```

**Content-Type**: `application/json`

| Parameter | Required | Type | Default | Description |
|------|------|------|--------|------|
| `sdp` | Yes | string | — | WebRTC Offer SDP |
| `type` | Yes | string | — | Must be `offer` |
| `avatar` | No | string | Startup parameter value | Specifies the avatar ID |
| `refaudio` | No | string | — | Reference audio |
| `reftext` | No | string | — | Reference text |
| `custom_config` | No | string | — | Action orchestration configuration JSON string |

**Response**:

```json
{
  "sdp": "v=0\r\n...",
  "type": "answer",
  "sessionid": "session-uuid"
}
```

---

## 2. Text-Driven (Human)

Send text to make the avatar speak, supporting either direct echo or LLM conversation.

```
POST /human
```

**Content-Type**: `application/json`

| Parameter | Required | Type | Default | Description |
|------|------|------|--------|------|
| `sessionid` | Yes | string | — | Session ID |
| `text` | Yes | string | — | Input text |
| `type` | Yes | string | — | `echo`: direct echo; `chat`: trigger an LLM response |
| `interrupt` | No | bool | false | Whether to interrupt the current playback |
| `tts` | No | object | — | Configuration passed through to TTS (e.g. `voice`, `emotion`) |

**Response**:

```json
{ "code": 0, "msg": "ok" }
```

---

## 3. Audio-Driven (Human Audio)

Upload an audio file to drive the avatar.

```
POST /humanaudio
```

**Content-Type**: `multipart/form-data`

| Parameter | Required | Type | Description |
|------|------|------|------|
| `sessionid` | Yes | string | Session ID |
| `file` | Yes | file | Audio file |

**Response**:

```json
{ "code": 0, "msg": "ok" }
```

---

## 4. Interrupt Playback

Immediately clear the current session's audio queue.

```
POST /interrupt_talk
```

| Parameter | Required | Type | Description |
|------|------|------|------|
| `sessionid` | Yes | string | Session ID |

**Response**:

```json
{ "code": 0, "msg": "ok" }
```

---

## 5. Query Speaking Status

```
POST /is_speaking
```

| Parameter | Required | Type | Description |
|------|------|------|------|
| `sessionid` | Yes | string | Session ID |

**Response**:

```json
{
  "code": 0,
  "msg": "ok",
  "data": true
}
```

---

## 6. Recording Control

Control server-side render recording.

```
POST /record
```

| Parameter | Required | Type | Description |
|------|------|------|------|
| `sessionid` | Yes | string | Session ID |
| `type` | Yes | string | `start_record`: start recording; `end_record`: stop and compose |

**Response**:

```json
{ "code": 0, "msg": "ok" }
```

---

## 7. Download Recording

Download the completed MP4 file.

```
GET /record/{sessionid}
```

**Path parameter**: `sessionid` — Session ID

**Response**: MP4 file stream. Returns 404 if the file does not exist.

---

## 8. Set Action Orchestration (Audiotype)

```
POST /set_audiotype
```

| Parameter | Required | Type | Description |
|------|------|------|------|
| `sessionid` | Yes | string | Session ID |
| `audiotype` | Yes | int | Predefined action/state index |

**Response**:

```json
{ "code": 0, "msg": "ok" }
```
