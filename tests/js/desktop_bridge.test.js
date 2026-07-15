const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

test("embedded Qt page never opens a duplicate websocket bridge", async () => {
  let websocketCount = 0;
  global.window = global;
  global.location = new URL("file:///C:/Local-AI-Companion/frontend/live2d/index.html");
  global.navigator = { userAgent: "QtWebEngine/6.8" };
  global.setTimeout = (callback) => callback();
  global.WebSocket = class {
    static OPEN = 1;

    constructor() {
      websocketCount += 1;
    }
  };

  require(path.resolve(__dirname, "../../frontend/live2d/desktop-bridge.js"));

  await assert.rejects(
    global.DesktopBridge.kind(),
    /Qt WebChannel did not become available/,
  );
  assert.equal(websocketCount, 0);
});
