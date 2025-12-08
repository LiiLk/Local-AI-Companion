/**
 * Live2D Manager for Local AI Companion
 * 
 * A modular, model-agnostic Live2D renderer that:
 * - Automatically detects expression parameters from .exp3.json files
 * - Never modifies parameters it shouldn't touch
 * - Works with any Live2D Cubism model (not just March 7th)
 * 
 * Design Philosophy:
 * - Only modify parameters that are explicitly defined in expression files
 * - Keep all other parameters at their model default values
 * - Provide clean APIs for lip-sync, head tracking, and expressions
 * 
 * Uses Cubism SDK for Web Core (live2dcubismcore.min.js)
 */

// Ensure Live2D Cubism Core is loaded
if (typeof Live2DCubismCore === 'undefined') {
    throw new Error('Live2DCubismCore is not loaded. Include live2dcubismcore.min.js first.');
}

/**
 * Live2D Manager Singleton
 */
const Live2DManager = (() => {
    // ==================== Private State ====================
    let _initialized = false;
    let _canvas = null;
    let _gl = null;
    let _model = null;
    let _modelSetting = null;
    let _textures = [];

    // Animation timing
    let _lastFrameTime = 0;
    let _deltaTime = 0;
    let _fps = 0;
    let _frameCount = 0;
    let _fpsUpdateTime = 0;

    // Model transform
    let _projectionMatrix = null;
    let _scale = 1.0;
    let _position = { x: 0, y: 0 };

    // Lip sync
    let _lipSyncValue = 0;
    let _targetLipSyncValue = 0;
    let _lipSyncSmoothing = 0.3;
    let _isLipSyncActive = false;

    // Eye blink state
    let _eyeBlinkState = {
        leftOpen: 1.0,
        rightOpen: 1.0,
        nextBlinkTime: 0,
        isBlinking: false,
        blinkProgress: 0
    };

    // Blink parameters from model (auto-detected)
    let _blinkParameterIds = [];

    // Lip sync parameters from model (auto-detected)
    let _lipSyncParameterIds = [];

    // Head tracking
    let _headAngle = { x: 0, y: 0, z: 0 };
    let _targetHeadAngle = { x: 0, y: 0, z: 0 };

    // Expression management - DYNAMICALLY POPULATED from .exp3.json files
    let _expressions = new Map();                    // name -> expression data
    let _currentExpression = null;                   // current expression name
    let _expressionParameterIds = new Set();         // Set of param IDs used by expressions (auto-detected)

    // Parameter cache (id -> index) for fast lookup
    let _parameterIndexCache = new Map();

    // Audio context for lip sync
    let _audioContext = null;
    let _analyser = null;
    let _audioSource = null;

    // Debug mode
    let _debugMode = false;
    let _config = {};

    // ==================== WebGL Initialization ====================

    function initWebGL(canvasId) {
        _canvas = document.getElementById(canvasId);
        if (!_canvas) {
            throw new Error(`Canvas element '${canvasId}' not found`);
        }

        _canvas.width = window.innerWidth;
        _canvas.height = window.innerHeight;

        const contextOptions = {
            alpha: true,
            premultipliedAlpha: true,
            antialias: true,
            stencil: true, // ERROR FIX: Enable stencil buffer for masking
            preserveDrawingBuffer: false,
            powerPreference: 'high-performance'
        };

        _gl = _canvas.getContext('webgl2', contextOptions) ||
            _canvas.getContext('webgl', contextOptions) ||
            _canvas.getContext('experimental-webgl', contextOptions);

        if (!_gl) {
            throw new Error('WebGL is not supported');
        }

        _gl.enable(_gl.BLEND);
        // Fix for "Transparent Face": Use premultiplied alpha blending
        _gl.blendFunc(_gl.ONE, _gl.ONE_MINUS_SRC_ALPHA);

        // Disable stencil test (we're not using masking)
        _gl.disable(_gl.STENCIL_TEST);

        _gl.clearColor(0.0, 0.0, 0.0, 0.0);

        window.addEventListener('resize', () => {
            _canvas.width = window.innerWidth;
            _canvas.height = window.innerHeight;
            _gl.viewport(0, 0, _canvas.width, _canvas.height);
            updateProjectionMatrix();
        });

        console.log('WebGL initialized:', _gl.getParameter(_gl.VERSION));
        return _gl;
    }

    // ==================== Model Loading ====================

    async function loadModelSetting(modelPath, modelName) {
        const settingPath = modelPath + modelName;
        console.log('Loading model setting from:', settingPath);

        const response = await fetch(settingPath);
        if (!response.ok) {
            throw new Error(`Failed to load model setting: ${response.status}`);
        }

        _modelSetting = await response.json();
        console.log('Model setting loaded:', _modelSetting);

        // Extract blink and lip sync parameter IDs from Groups
        if (_modelSetting.Groups) {
            for (const group of _modelSetting.Groups) {
                if (group.Name === 'EyeBlink' && group.Ids) {
                    _blinkParameterIds = [...group.Ids];
                    console.log('Eye blink parameters:', _blinkParameterIds);
                }
                if (group.Name === 'LipSync' && group.Ids) {
                    _lipSyncParameterIds = [...group.Ids];
                    console.log('Lip sync parameters:', _lipSyncParameterIds);
                }
            }
        }

        // Default lip sync parameter if not specified
        if (_lipSyncParameterIds.length === 0) {
            _lipSyncParameterIds = ['ParamMouthOpenY'];
        }

        return _modelSetting;
    }

    async function loadMoc(modelPath) {
        const mocPath = modelPath + _modelSetting.FileReferences.Moc;
        console.log('Loading MOC3 from:', mocPath);

        const response = await fetch(mocPath);
        if (!response.ok) {
            throw new Error(`Failed to load MOC3: ${response.status}`);
        }

        const arrayBuffer = await response.arrayBuffer();

        const moc = Live2DCubismCore.Moc.fromArrayBuffer(arrayBuffer);
        if (!moc) {
            throw new Error('Failed to create MOC from file');
        }

        _model = Live2DCubismCore.Model.fromMoc(moc);
        if (!_model) {
            throw new Error('Failed to create model from MOC');
        }

        console.log('Model loaded successfully');
        console.log('Parameter count:', _model.parameters.count);
        console.log('Part count:', _model.parts.count);
        console.log('Drawable count:', _model.drawables.count);

        // Build parameter index cache for fast lookup
        _parameterIndexCache.clear();
        for (let i = 0; i < _model.parameters.count; i++) {
            _parameterIndexCache.set(_model.parameters.ids[i], i);
        }

        return _model;
    }

    async function loadTextures(modelPath) {
        const textureFiles = _modelSetting.FileReferences.Textures;
        _textures = [];

        for (let i = 0; i < textureFiles.length; i++) {
            const texturePath = modelPath + textureFiles[i];
            console.log('Loading texture:', texturePath);

            const texture = await loadTexture(texturePath);
            _textures.push(texture);
        }

        console.log(`Loaded ${_textures.length} textures`);
        return _textures;
    }

    function loadTexture(path) {
        return new Promise((resolve, reject) => {
            const image = new Image();
            image.crossOrigin = 'anonymous';

            image.onload = () => {
                const texture = _gl.createTexture();
                _gl.bindTexture(_gl.TEXTURE_2D, texture);

                _gl.pixelStorei(_gl.UNPACK_FLIP_Y_WEBGL, true);
                _gl.pixelStorei(_gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);

                _gl.texImage2D(_gl.TEXTURE_2D, 0, _gl.RGBA, _gl.RGBA, _gl.UNSIGNED_BYTE, image);

                _gl.texParameteri(_gl.TEXTURE_2D, _gl.TEXTURE_MIN_FILTER, _gl.LINEAR_MIPMAP_LINEAR);
                _gl.texParameteri(_gl.TEXTURE_2D, _gl.TEXTURE_MAG_FILTER, _gl.LINEAR);
                _gl.texParameteri(_gl.TEXTURE_2D, _gl.TEXTURE_WRAP_S, _gl.CLAMP_TO_EDGE);
                _gl.texParameteri(_gl.TEXTURE_2D, _gl.TEXTURE_WRAP_T, _gl.CLAMP_TO_EDGE);

                _gl.generateMipmap(_gl.TEXTURE_2D);
                _gl.bindTexture(_gl.TEXTURE_2D, null);

                resolve({ texture, width: image.width, height: image.height });
            };

            image.onerror = () => reject(new Error(`Failed to load texture: ${path}`));
            image.src = path;
        });
    }

    // ==================== Expression Loading ====================

    /**
     * Load expressions from .exp3.json files
     * 
     * IMPORTANT: This dynamically detects which parameters are used by expressions
     * so we know exactly which ones to reset when changing expressions.
     */
    async function loadExpressions(modelPath) {
        const expressionDefs = _modelSetting.FileReferences.Expressions || [];
        _expressions.clear();
        _expressionParameterIds.clear();

        for (const expDef of expressionDefs) {
            try {
                // Expression files are in the 'exp/' subdirectory
                const expPath = modelPath + 'exp/' + expDef.File;
                const response = await fetch(expPath);
                if (response.ok) {
                    const expData = await response.json();
                    _expressions.set(expDef.Name, expData);

                    // CRITICAL: Track which parameters are used by expressions
                    // We will ONLY reset these parameters when changing expressions
                    if (expData.Parameters) {
                        for (const param of expData.Parameters) {
                            _expressionParameterIds.add(param.Id);
                        }
                    }

                    console.log('Loaded expression:', expDef.Name,
                        '-> Params:', expData.Parameters?.map(p => p.Id).join(', ') || 'none');
                }
            } catch (error) {
                console.warn(`Failed to load expression ${expDef.Name}:`, error);
            }
        }

        console.log('=== Expression System ===');
        console.log('Available expressions:', Array.from(_expressions.keys()));
        console.log('Expression parameters (will be reset on change):', Array.from(_expressionParameterIds));

        return _expressions;
    }

    /**
     * Reset ONLY the parameters that are used by expressions
     * 
     * This is the key to not breaking the model's face:
     * - We only touch parameters that expressions actually modify
     * - All other parameters stay at their model defaults
     */
    function resetExpressionParameters() {
        if (!_model) return;

        for (const paramId of _expressionParameterIds) {
            const index = _parameterIndexCache.get(paramId);
            if (index !== undefined) {
                // Reset to default value (NOT 0!)
                // If we reset to 0, eyes will close (because default is 1.0)
                const defaultVal = _model.parameters.defaultValues[index];
                _model.parameters.values[index] = defaultVal;
            }
        }

        if (_debugMode) {
            console.log('Reset expression parameters:', Array.from(_expressionParameterIds));
        }
    }

    /**
     * Apply an expression by name
     */
    function applyExpression(name) {
        if (!_model) {
            console.warn('Cannot apply expression: model not loaded');
            return false;
        }

        // Build name mapping for this model's expressions
        const expressionNameMap = buildExpressionNameMap();

        // Normalize and lookup
        const normalizedName = name.toLowerCase().trim();
        const mappedName = expressionNameMap[normalizedName] ?? name;

        // STEP 1: Reset ONLY expression parameters to 0
        resetExpressionParameters();

        // STEP 2: If neutral/null, we're done
        if (mappedName === null || mappedName === 'neutral') {
            _currentExpression = 'neutral';
            console.log('Expression reset to neutral');
            updateDebugDisplay();
            return true;
        }

        // STEP 3: Find and apply the expression
        const expression = _expressions.get(mappedName);
        if (!expression) {
            console.warn(`Expression not found: "${name}" (mapped: "${mappedName}")`);
            console.log('Available:', Array.from(_expressions.keys()));
            return false;
        }

        // STEP 4: Apply expression parameters
        if (expression.Parameters) {
            for (const param of expression.Parameters) {
                const index = _parameterIndexCache.get(param.Id);
                if (index !== undefined) {
                    const blendMode = param.Blend || 'Override';

                    if (blendMode === 'Add') {
                        _model.parameters.values[index] += param.Value;
                    } else if (blendMode === 'Multiply') {
                        _model.parameters.values[index] *= param.Value;
                    } else {
                        _model.parameters.values[index] = param.Value;
                    }
                }
            }
        }

        _currentExpression = mappedName;
        console.log('Expression applied:', mappedName);
        updateDebugDisplay();

        return true;
    }

    /**
     * Build expression name mapping dynamically based on loaded expressions
     */
    function buildExpressionNameMap() {
        const map = {
            'neutral': null,
            'default': null,
            'normal': null,
            'reset': null,
        };

        // Add direct mappings for all loaded expressions
        for (const name of _expressions.keys()) {
            map[name] = name;
            map[name.toLowerCase()] = name;
        }

        // Add common English aliases if expressions exist
        const aliases = {
            // Chinese -> English aliases
            '捂脸': ['facecover', 'cover', 'facepalm'],
            '比耶': ['peace', 'victory', 'yeah', 'v'],
            '照相': ['photo', 'camera', 'picture'],
            '脸红': ['blush', 'shy', 'embarrassed'],
            '黑脸': ['angry', 'dark', 'mad', 'annoyed'],
            '哭': ['cry', 'sad', 'tears', 'crying'],
            '流汗': ['sweat', 'nervous', 'anxious', 'worried'],
            '星星': ['stars', 'happy', 'excited', 'sparkle', 'joy'],
        };

        for (const [chinese, englishList] of Object.entries(aliases)) {
            if (_expressions.has(chinese)) {
                for (const eng of englishList) {
                    map[eng] = chinese;
                }
            }
        }

        return map;
    }

    function updateDebugDisplay() {
        if (_debugMode) {
            const expEl = document.getElementById('expression-name');
            if (expEl) expEl.textContent = _currentExpression || 'neutral';
        }
    }

    async function loadPhysics(modelPath) {
        const physicsFile = _modelSetting.FileReferences.Physics;
        if (!physicsFile) return null;

        try {
            const physicsPath = modelPath + physicsFile;
            const response = await fetch(physicsPath);
            if (response.ok) {
                console.log('Physics loaded');
                return await response.json();
            }
        } catch (error) {
            console.warn('Failed to load physics:', error);
        }
        return null;
    }

    // ==================== Parameter Access ====================

    function setParameter(paramId, value) {
        if (!_model) return;

        const index = _parameterIndexCache.get(paramId);
        if (index !== undefined) {
            const minVal = _model.parameters.minimumValues[index];
            const maxVal = _model.parameters.maximumValues[index];
            _model.parameters.values[index] = Math.max(minVal, Math.min(maxVal, value));
        }
    }

    function getParameter(paramId) {
        if (!_model) return 0;

        const index = _parameterIndexCache.get(paramId);
        return index !== undefined ? _model.parameters.values[index] : 0;
    }

    // ==================== Animation Updates ====================

    function updateProjectionMatrix() {
        const aspect = _canvas.width / _canvas.height;

        _projectionMatrix = new Float32Array([
            _scale / aspect, 0, 0, 0,
            0, _scale, 0, 0,
            0, 0, 1, 0,
            _position.x, _position.y, 0, 1
        ]);
    }

    function updateEyeBlink(deltaTime) {
        if (_blinkParameterIds.length === 0) return;

        const now = performance.now() / 1000;

        if (!_eyeBlinkState.isBlinking) {
            if (now >= _eyeBlinkState.nextBlinkTime) {
                _eyeBlinkState.isBlinking = true;
                _eyeBlinkState.blinkProgress = 0;
            }
        }

        if (_eyeBlinkState.isBlinking) {
            _eyeBlinkState.blinkProgress += deltaTime * 8;

            let openValue;
            if (_eyeBlinkState.blinkProgress < 0.5) {
                openValue = 1 - (_eyeBlinkState.blinkProgress * 2);
            } else if (_eyeBlinkState.blinkProgress < 1.0) {
                openValue = (_eyeBlinkState.blinkProgress - 0.5) * 2;
            } else {
                _eyeBlinkState.isBlinking = false;
                openValue = 1;
                _eyeBlinkState.nextBlinkTime = now + 2 + Math.random() * 4;
            }

            // Apply to all blink parameters
            for (const paramId of _blinkParameterIds) {
                setParameter(paramId, openValue);
            }
        }
    }

    function updateBreathing(deltaTime) {
        const time = performance.now() / 1000;
        const breathValue = (Math.sin(time * 2) + 1) * 0.5;

        setParameter('ParamBreath', breathValue);

        // Subtle body sway
        const bodyAngle = Math.sin(time * 2) * 2;
        setParameter('ParamBodyAngleY', bodyAngle);
    }

    function updateLipSync(deltaTime) {
        _lipSyncValue += (_targetLipSyncValue - _lipSyncValue) * _lipSyncSmoothing;

        // Apply to all lip sync parameters
        for (const paramId of _lipSyncParameterIds) {
            setParameter(paramId, _lipSyncValue);
        }

        if (_debugMode) {
            const lipEl = document.getElementById('lip-value');
            if (lipEl) lipEl.textContent = _lipSyncValue.toFixed(2);
        }
    }

    function updateHeadTracking(deltaTime) {
        const smoothing = 0.1;

        _headAngle.x += (_targetHeadAngle.x - _headAngle.x) * smoothing;
        _headAngle.y += (_targetHeadAngle.y - _headAngle.y) * smoothing;
        _headAngle.z += (_targetHeadAngle.z - _headAngle.z) * smoothing;

        setParameter('ParamAngleX', _headAngle.x);
        setParameter('ParamAngleY', _headAngle.y);
        setParameter('ParamAngleZ', _headAngle.z);

        // Eye follow (scaled down)
        setParameter('ParamEyeBallX', _headAngle.x / 30);
        setParameter('ParamEyeBallY', _headAngle.y / 30);
    }

    function update(timestamp) {
        _deltaTime = (timestamp - _lastFrameTime) / 1000;
        _lastFrameTime = timestamp;

        if (_deltaTime > 0.1) _deltaTime = 0.1;

        _frameCount++;
        if (timestamp - _fpsUpdateTime >= 1000) {
            _fps = _frameCount;
            _frameCount = 0;
            _fpsUpdateTime = timestamp;

            if (_debugMode) {
                const fpsEl = document.getElementById('fps');
                if (fpsEl) fpsEl.textContent = _fps;
            }
        }

        if (_model) {
            updateEyeBlink(_deltaTime);
            updateBreathing(_deltaTime);
            updateLipSync(_deltaTime);
            updateHeadTracking(_deltaTime);

            _model.update();
            render();
        }

        requestAnimationFrame(update);
    }

    // ==================== WebGL Rendering ====================

    let _shaderProgram = null;
    let _shaderLocations = {};

    function initShaders() {
        const vertexShaderSource = `
            attribute vec2 a_position;
            attribute vec2 a_texCoord;
            uniform mat4 u_matrix;
            varying vec2 v_texCoord;
            
            void main() {
                gl_Position = u_matrix * vec4(a_position, 0.0, 1.0);
                v_texCoord = a_texCoord;
            }
        `;

        const fragmentShaderSource = `
            precision mediump float;
            varying vec2 v_texCoord;
            uniform sampler2D u_texture;
            uniform float u_opacity;
            uniform vec4 u_baseColor;
            
            void main() {
                vec4 texColor = texture2D(u_texture, v_texCoord);
                
                // For premultiplied alpha, we output the color directly
                // The opacity is already in the alpha channel from the texture
                gl_FragColor = texColor * u_baseColor * u_opacity;
            }
        `;

        const vertexShader = compileShader(_gl.VERTEX_SHADER, vertexShaderSource);
        const fragmentShader = compileShader(_gl.FRAGMENT_SHADER, fragmentShaderSource);

        _shaderProgram = _gl.createProgram();
        _gl.attachShader(_shaderProgram, vertexShader);
        _gl.attachShader(_shaderProgram, fragmentShader);
        _gl.linkProgram(_shaderProgram);

        if (!_gl.getProgramParameter(_shaderProgram, _gl.LINK_STATUS)) {
            throw new Error('Shader program link failed: ' + _gl.getProgramInfoLog(_shaderProgram));
        }

        _shaderLocations = {
            position: _gl.getAttribLocation(_shaderProgram, 'a_position'),
            texCoord: _gl.getAttribLocation(_shaderProgram, 'a_texCoord'),
            matrix: _gl.getUniformLocation(_shaderProgram, 'u_matrix'),
            texture: _gl.getUniformLocation(_shaderProgram, 'u_texture'),
            opacity: _gl.getUniformLocation(_shaderProgram, 'u_opacity'),
            baseColor: _gl.getUniformLocation(_shaderProgram, 'u_baseColor')
        };

        console.log('Shaders initialized');
    }

    function compileShader(type, source) {
        const shader = _gl.createShader(type);
        _gl.shaderSource(shader, source);
        _gl.compileShader(shader);

        if (!_gl.getShaderParameter(shader, _gl.COMPILE_STATUS)) {
            const info = _gl.getShaderInfoLog(shader);
            _gl.deleteShader(shader);
            throw new Error('Shader compile failed: ' + info);
        }

        return shader;
    }

    function render() {
        // Clear Color
        _gl.clear(_gl.COLOR_BUFFER_BIT);

        if (!_model || !_shaderProgram || _textures.length === 0) return;

        _gl.useProgram(_shaderProgram);

        const drawableCount = _model.drawables.count;
        const renderOrders = _model.drawables.renderOrders;
        const opacities = _model.drawables.opacities;
        const textureIndices = _model.drawables.textureIndices;
        const vertexPositions = _model.drawables.vertexPositions;
        const vertexUvs = _model.drawables.vertexUvs;
        const indices = _model.drawables.indices;

        // Sort by render order
        const sortedIndices = [];
        for (let i = 0; i < drawableCount; i++) {
            sortedIndices.push(i);
        }
        sortedIndices.sort((a, b) => renderOrders[a] - renderOrders[b]);

        // Debug: Log once how many drawables we have
        if (!_debugLogged) {
            _debugLogged = true;
            console.log(`=== Render Debug ===`);
            console.log(`Total drawables: ${drawableCount}`);
            console.log(`Textures loaded: ${_textures.length}`);
        }

        // Draw each drawable in order (simple approach without masking)
        for (const drawableIndex of sortedIndices) {
            const opacity = opacities[drawableIndex];
            if (opacity <= 0) continue;

            const textureIndex = textureIndices[drawableIndex];
            if (textureIndex < 0 || textureIndex >= _textures.length) continue;

            const positions = vertexPositions[drawableIndex];
            const uvs = vertexUvs[drawableIndex];
            const indexArray = indices[drawableIndex];

            if (!positions || positions.length === 0 || !indexArray || indexArray.length === 0) continue;

            // Bind texture
            _gl.activeTexture(_gl.TEXTURE0);
            _gl.bindTexture(_gl.TEXTURE_2D, _textures[textureIndex].texture);
            _gl.uniform1i(_shaderLocations.texture, 0);

            // Set uniforms
            _gl.uniformMatrix4fv(_shaderLocations.matrix, false, _projectionMatrix);
            _gl.uniform1f(_shaderLocations.opacity, opacity);
            _gl.uniform4f(_shaderLocations.baseColor, 1.0, 1.0, 1.0, 1.0);

            // Create and bind position buffer
            const positionBuffer = _gl.createBuffer();
            _gl.bindBuffer(_gl.ARRAY_BUFFER, positionBuffer);
            _gl.bufferData(_gl.ARRAY_BUFFER, positions, _gl.DYNAMIC_DRAW);
            _gl.enableVertexAttribArray(_shaderLocations.position);
            _gl.vertexAttribPointer(_shaderLocations.position, 2, _gl.FLOAT, false, 0, 0);

            // Create and bind UV buffer
            const uvBuffer = _gl.createBuffer();
            _gl.bindBuffer(_gl.ARRAY_BUFFER, uvBuffer);
            _gl.bufferData(_gl.ARRAY_BUFFER, uvs, _gl.STATIC_DRAW);
            _gl.enableVertexAttribArray(_shaderLocations.texCoord);
            _gl.vertexAttribPointer(_shaderLocations.texCoord, 2, _gl.FLOAT, false, 0, 0);

            // Create and bind index buffer
            const indexBuffer = _gl.createBuffer();
            _gl.bindBuffer(_gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
            _gl.bufferData(_gl.ELEMENT_ARRAY_BUFFER, indexArray, _gl.STATIC_DRAW);

            // Draw
            _gl.drawElements(_gl.TRIANGLES, indexArray.length, _gl.UNSIGNED_SHORT, 0);

            // Cleanup buffers
            _gl.deleteBuffer(positionBuffer);
            _gl.deleteBuffer(uvBuffer);
            _gl.deleteBuffer(indexBuffer);
        }
    }

    let _debugLogged = false;

    // ==================== Input Handling ====================

    function setupMouseTracking() {
        document.addEventListener('mousemove', (e) => {
            const rect = _canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width * 2 - 1;
            const y = -((e.clientY - rect.top) / rect.height * 2 - 1);

            _targetHeadAngle.x = x * 30;
            _targetHeadAngle.y = y * 30;
        });
    }

    // ==================== Audio Analysis ====================

    async function setupAudioAnalysis() {
        try {
            _audioContext = new (window.AudioContext || window.webkitAudioContext)();
            _analyser = _audioContext.createAnalyser();
            _analyser.fftSize = 256;
            _analyser.smoothingTimeConstant = 0.5;
            console.log('Audio analysis ready');
        } catch (error) {
            console.warn('Failed to set up audio analysis:', error);
        }
    }

    function analyzeAudioForLipSync() {
        if (!_analyser || !_isLipSyncActive) return;

        const dataArray = new Uint8Array(_analyser.frequencyBinCount);
        _analyser.getByteFrequencyData(dataArray);

        let sum = 0;
        for (let i = 2; i < 24 && i < dataArray.length; i++) {
            sum += dataArray[i];
        }

        const average = sum / 22;
        let normalizedValue = Math.min(1, Math.max(0, (average / 128 - 0.1) * 1.5));

        _targetLipSyncValue = normalizedValue;

        if (_isLipSyncActive) {
            requestAnimationFrame(analyzeAudioForLipSync);
        }
    }

    // ==================== Public API ====================

    return {
        /**
         * Initialize Live2D manager
         */
        async init(config) {
            _config = config;
            _debugMode = config.debug || false;
            _scale = config.scale || 1.0;
            _position = config.position || { x: 0, y: 0 };

            try {
                initWebGL(config.canvasId);
                initShaders();

                await loadModelSetting(config.modelPath, config.modelName);
                await loadMoc(config.modelPath);
                await loadTextures(config.modelPath);
                await loadExpressions(config.modelPath);
                await loadPhysics(config.modelPath);

                // DO NOT reset any parameters here!
                // The model loads with correct default values from the .moc3 file

                // CRITICAL FIX: Reset ALL parameters to their default values from .moc3
                // The SDK sometimes doesn't initialize them correctly
                console.log('=== Resetting ALL parameters to .moc3 defaults ===');
                for (let i = 0; i < _model.parameters.count; i++) {
                    const defaultVal = _model.parameters.defaultValues[i];
                    _model.parameters.values[i] = defaultVal;
                }

                // DEBUG: Log critical eye parameters after reset
                console.log('=== Parameter Values After Reset ===');
                const criticalParams = [
                    'Param', 'Param2', 'Param3', 'Param4', 'Param5', 'Param6', 'Param7',
                    'ParamEyeLOpen', 'ParamEyeROpen', 'ParamEyeBallX', 'ParamEyeBallY'
                ];
                for (const paramId of criticalParams) {
                    const index = _parameterIndexCache.get(paramId);
                    if (index !== undefined) {
                        console.log(`  ${paramId}: value=${_model.parameters.values[index].toFixed(3)}, default=${_model.parameters.defaultValues[index]}, range=[${_model.parameters.minimumValues[index]}, ${_model.parameters.maximumValues[index]}]`);
                    }
                }

                if (_debugMode) {
                    const modelNameEl = document.getElementById('model-name');
                    if (modelNameEl) modelNameEl.textContent = config.modelName;
                }

                updateProjectionMatrix();
                setupMouseTracking();
                await setupAudioAnalysis();

                _eyeBlinkState.nextBlinkTime = performance.now() / 1000 + 2;

                // Apply default expression if configured
                if (config.defaultExpression && config.defaultExpression !== 'neutral') {
                    setTimeout(() => this.setExpression(config.defaultExpression), 100);
                }

                _lastFrameTime = performance.now();
                requestAnimationFrame(update);

                _initialized = true;
                console.log('Live2D Manager initialized successfully');
                console.log('Model is ready with default appearance from .moc3');

            } catch (error) {
                console.error('Failed to initialize Live2D:', error);
                throw error;
            }
        },

        // Lip sync
        setLipSync(value) {
            _targetLipSyncValue = Math.max(0, Math.min(1, value));
        },

        async startAudioLipSync() {
            if (!_audioContext || !_analyser) return;
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                _audioSource = _audioContext.createMediaStreamSource(stream);
                _audioSource.connect(_analyser);
                _isLipSyncActive = true;
                analyzeAudioForLipSync();
                console.log('Audio lip sync started');
            } catch (error) {
                console.warn('Failed to start audio lip sync:', error);
            }
        },

        connectAudioElement(audioElement) {
            if (!_audioContext || !_analyser) return;
            try {
                _audioSource = _audioContext.createMediaElementSource(audioElement);
                _audioSource.connect(_analyser);
                _audioSource.connect(_audioContext.destination);
                _isLipSyncActive = true;
                analyzeAudioForLipSync();
                console.log('Audio element connected');
            } catch (error) {
                console.warn('Failed to connect audio element:', error);
            }
        },

        stopLipSync() {
            _isLipSyncActive = false;
            _targetLipSyncValue = 0;
            if (_audioSource) {
                _audioSource.disconnect();
                _audioSource = null;
            }
        },

        // Expressions
        setExpression(name) {
            return applyExpression(name);
        },

        resetExpression() {
            return applyExpression('neutral');
        },

        getExpressionNames() {
            return Array.from(_expressions.keys());
        },

        getCurrentExpression() {
            return _currentExpression;
        },

        // Parameters
        setParameter(paramId, value) {
            setParameter(paramId, value);
        },

        getParameter(paramId) {
            return getParameter(paramId);
        },

        /**
         * List all parameters for debugging
         */
        listAllParameters() {
            if (!_model) return [];

            console.log('=== All Model Parameters ===');
            const params = [];
            for (let i = 0; i < _model.parameters.count; i++) {
                const param = {
                    id: _model.parameters.ids[i],
                    value: _model.parameters.values[i],
                    default: _model.parameters.defaultValues[i],
                    min: _model.parameters.minimumValues[i],
                    max: _model.parameters.maximumValues[i]
                };
                params.push(param);
                console.log(`${param.id}: ${param.value.toFixed(3)} (default: ${param.default})`);
            }
            return params;
        },

        /**
         * List expression parameters only
         */
        listExpressionParameters() {
            console.log('=== Expression Parameters ===');
            console.log('These are the ONLY parameters that will be modified by expressions:');
            for (const paramId of _expressionParameterIds) {
                const value = getParameter(paramId);
                console.log(`  ${paramId}: ${value}`);
            }
            return Array.from(_expressionParameterIds);
        },

        // Transform
        setHeadAngle(x, y, z = 0) {
            _targetHeadAngle.x = x;
            _targetHeadAngle.y = y;
            _targetHeadAngle.z = z;
        },

        setPosition(x, y) {
            _position.x = x;
            _position.y = y;
            updateProjectionMatrix();
        },

        setScale(scale) {
            _scale = scale;
            updateProjectionMatrix();
        },

        // Motions (stub)
        playMotion(group, index) {
            console.log('Play motion:', group, index);
        },

        getMotionGroups() {
            return [];
        },

        // Utility
        toggleDebug() {
            _debugMode = !_debugMode;
            return _debugMode;
        },

        isInitialized() {
            return _initialized;
        },

        getFPS() {
            return _fps;
        }
    };
})();

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Live2DManager;
}
