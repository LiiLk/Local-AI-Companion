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

test("stale decode after stop does not clear new playback", async () => {
  const started = [];
  let resolveFirst;
  global.window = global;
  global.requestAnimationFrame = () => 0;
  global.Live2DManager = { setLipSync() {}, setExpression() {} };
  global.AudioContext = class extends FakeAudioContext {
    constructor() {
      super(started);
      this.decodeCount = 0;
    }

    decodeAudioData() {
      this.decodeCount += 1;
      if (this.decodeCount === 1) {
        return new Promise((resolve) => { resolveFirst = resolve; });
      }
      return Promise.resolve({ duration: 0.25 });
    }
  };

  delete require.cache[require.resolve(path.resolve(__dirname, "../../frontend/js/live2d-integration.js"))];
  require(path.resolve(__dirname, "../../frontend/js/live2d-integration.js"));
  const integration = new global.Live2DIntegration();
  integration.ready = true;

  integration.handleAudioMessage({ data: new Uint8Array([1]).buffer });
  await nextTurn();
  integration.stopPlayback();
  integration.handleAudioMessage({ data: new Uint8Array([2]).buffer });
  await nextTurn();
  assert.equal(started.length, 1);
  assert.equal(integration._playing, true);

  resolveFirst({ duration: 0.25 });
  await nextTurn();
  assert.equal(started.length, 1);
  assert.equal(integration._playing, true);
});

test("ended chunk lip-sync loop stops when queued chunk starts", async () => {
  const started = [];
  const frameCallbacks = [];
  let lipWrites = 0;
  global.window = global;
  global.requestAnimationFrame = (callback) => { frameCallbacks.push(callback); return frameCallbacks.length; };
  global.Live2DManager = { setLipSync() { lipWrites += 1; }, setExpression() {} };
  global.AudioContext = class extends FakeAudioContext {
    constructor() {
      super(started);
    }
  };

  delete require.cache[require.resolve(path.resolve(__dirname, "../../frontend/js/live2d-integration.js"))];
  require(path.resolve(__dirname, "../../frontend/js/live2d-integration.js"));
  const integration = new global.Live2DIntegration();
  integration.ready = true;

  integration.handleAudioMessage({ data: new Uint8Array([1]).buffer });
  integration.handleAudioMessage({ data: new Uint8Array([2]).buffer });
  await nextTurn();
  const oldTick = frameCallbacks[0];
  started[0].onended();
  await nextTurn();
  const before = lipWrites;
  oldTick();
  assert.equal(lipWrites, before);
});
