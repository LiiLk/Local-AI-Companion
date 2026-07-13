const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

class FakeSource {
  constructor(started) {
    this.started = started;
    this.onended = null;
    this.stopped = false;
  }

  connect() {}

  start() {
    this.started.push(this);
  }

  stop() {
    this.stopped = true;
  }
}

class FakeAudioContext {
  constructor(started) {
    this.started = started;
    this.state = "running";
  }

  decodeAudioData() {
    return Promise.resolve({ duration: 0.25 });
  }

  createBufferSource() {
    return new FakeSource(this.started);
  }

  createAnalyser() {
    return {
      connect() {},
      frequencyBinCount: 8,
      getByteFrequencyData() {},
    };
  }
}

function nextTurn() {
  return new Promise((resolve) => setImmediate(resolve));
}

test("Live2D TTS chunks play sequentially", async () => {
  const started = [];
  global.window = global;
  global.requestAnimationFrame = () => 0;
  global.Live2DManager = { setLipSync() {}, setExpression() {} };
  global.AudioContext = class extends FakeAudioContext {
    constructor() {
      super(started);
    }
  };

  require(path.resolve(__dirname, "../../frontend/js/live2d-integration.js"));
  const integration = new global.Live2DIntegration();
  integration.ready = true;
  let starts = 0;
  let ends = 0;
  integration.onPlaybackStart = () => { starts += 1; };
  integration.onPlaybackEnd = () => { ends += 1; };

  integration.handleAudioMessage({ data: new Uint8Array([1]).buffer });
  integration.handleAudioMessage({ data: new Uint8Array([2]).buffer });
  await nextTurn();

  assert.equal(started.length, 1);
  assert.equal(started[0].stopped, false);
  assert.equal(starts, 1);
  assert.equal(ends, 0);

  started[0].onended();
  await nextTurn();
  assert.equal(started.length, 2);
  assert.equal(started[1].stopped, false);

  started[1].onended();
  assert.equal(starts, 1);
  assert.equal(ends, 1);
});
