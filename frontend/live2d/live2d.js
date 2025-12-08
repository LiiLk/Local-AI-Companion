/**
 * Live2D Manager for Local AI Companion
 * 
 * Handles Live2D Cubism SDK initialization, model loading, rendering,
 * and provides APIs for lip-sync, expressions, and motions.
 * 
 * Uses Cubism SDK for Web (Core + Framework concepts reimplemented in vanilla JS)
 */

// Ensure Live2D Cubism Core is loaded
if (typeof Live2DCubismCore === 'undefined') {
    throw new Error('Live2DCubismCore is not loaded. Include live2dcubismcore.min.js first.');
}

/**
 * Live2D Manager Singleton
 */
const Live2DManager = (() => {
    // Private state
    let _initialized = false;
    let _canvas = null;
    let _gl = null;
    let _model = null;
    let _modelSetting = null;
    let _textures = [];
    let _frameBuffer = null;
    
    // Animation state
    let _lastFrameTime = 0;
    let _deltaTime = 0;
    let _fps = 0;
    let _frameCount = 0;
    let _fpsUpdateTime = 0;
    
    // Model transform
    let _modelMatrix = null;
    let _projectionMatrix = null;
    let _scale = 1.0;
    let _position = { x: 0, y: 0 };
    
    // Parameters for animation
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
    
    // Breathing animation
    let _breathValue = 0;
    
    // Head tracking (for mouse follow)
    let _headAngle = { x: 0, y: 0, z: 0 };
    let _targetHeadAngle = { x: 0, y: 0, z: 0 };
    
    // Expression state
    let _currentExpression = null;
    let _expressions = new Map();
    
    // Motion state  
    let _motions = new Map();
    let _currentMotion = null;
    
    // Audio context for lip sync
    let _audioContext = null;
    let _analyser = null;
    let _audioSource = null;
    
    // Debug mode
    let _debugMode = false;
    let _config = {};
    
    // Parameter IDs (will be populated from model)
    const PARAM_IDS = {
        // Standard Cubism parameters
        angleX: 'ParamAngleX',
        angleY: 'ParamAngleY', 
        angleZ: 'ParamAngleZ',
        eyeLOpen: 'ParamEyeLOpen',
        eyeROpen: 'ParamEyeROpen',
        eyeBallX: 'ParamEyeBallX',
        eyeBallY: 'ParamEyeBallY',
        mouthOpenY: 'ParamMouthOpenY',
        mouthForm: 'ParamMouthForm',
        bodyAngleX: 'ParamBodyAngleX',
        bodyAngleY: 'ParamBodyAngleY',
        bodyAngleZ: 'ParamBodyAngleZ',
        breath: 'ParamBreath',
        // March 7th specific - may need adjustment
        browLY: 'ParamBrowLY',
        browRY: 'ParamBrowRY'
    };
    
    /**
     * Initialize WebGL context
     */
    function initWebGL(canvasId) {
        _canvas = document.getElementById(canvasId);
        if (!_canvas) {
            throw new Error(`Canvas element '${canvasId}' not found`);
        }
        
        // Set canvas size to window size
        _canvas.width = window.innerWidth;
        _canvas.height = window.innerHeight;
        
        // Get WebGL context with transparency
        const contextOptions = {
            alpha: true,
            premultipliedAlpha: true,
            antialias: true,
            preserveDrawingBuffer: false,
            powerPreference: 'high-performance'
        };
        
        _gl = _canvas.getContext('webgl2', contextOptions) || 
              _canvas.getContext('webgl', contextOptions) ||
              _canvas.getContext('experimental-webgl', contextOptions);
              
        if (!_gl) {
            throw new Error('WebGL is not supported');
        }
        
        // Enable blending for transparency
        _gl.enable(_gl.BLEND);
        _gl.blendFunc(_gl.SRC_ALPHA, _gl.ONE_MINUS_SRC_ALPHA);
        
        // Clear with transparent color
        _gl.clearColor(0.0, 0.0, 0.0, 0.0);
        
        // Handle window resize
        window.addEventListener('resize', () => {
            _canvas.width = window.innerWidth;
            _canvas.height = window.innerHeight;
            _gl.viewport(0, 0, _canvas.width, _canvas.height);
            updateProjectionMatrix();
        });
        
        console.log('WebGL initialized:', _gl.getParameter(_gl.VERSION));
        return _gl;
    }
    
    /**
     * Load model settings from model3.json
     */
    async function loadModelSetting(modelPath, modelName) {
        const settingPath = modelPath + modelName;
        console.log('Loading model setting from:', settingPath);
        
        const response = await fetch(settingPath);
        if (!response.ok) {
            throw new Error(`Failed to load model setting: ${response.status}`);
        }
        
        _modelSetting = await response.json();
        console.log('Model setting loaded:', _modelSetting);
        
        return _modelSetting;
    }
    
    /**
     * Load MOC3 model file
     */
    async function loadMoc(modelPath) {
        const mocPath = modelPath + _modelSetting.FileReferences.Moc;
        console.log('Loading MOC3 from:', mocPath);
        
        const response = await fetch(mocPath);
        if (!response.ok) {
            throw new Error(`Failed to load MOC3: ${response.status}`);
        }
        
        const arrayBuffer = await response.arrayBuffer();
        
        // Create MOC using Cubism Core
        const moc = Live2DCubismCore.Moc.fromArrayBuffer(arrayBuffer);
        if (!moc) {
            throw new Error('Failed to create MOC from file');
        }
        
        // Create model from MOC
        _model = Live2DCubismCore.Model.fromMoc(moc);
        if (!_model) {
            throw new Error('Failed to create model from MOC');
        }
        
        console.log('Model loaded successfully');
        console.log('Parameter count:', _model.parameters.count);
        console.log('Part count:', _model.parts.count);
        console.log('Drawable count:', _model.drawables.count);
        
        // Log available parameters for debugging
        if (_debugMode) {
            console.log('Available parameters:');
            for (let i = 0; i < _model.parameters.count; i++) {
                console.log(`  ${_model.parameters.ids[i]}: ${_model.parameters.defaultValues[i]} [${_model.parameters.minimumValues[i]}, ${_model.parameters.maximumValues[i]}]`);
            }
        }
        
        return _model;
    }
    
    /**
     * Load textures
     */
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
    
    /**
     * Load a single texture
     */
    function loadTexture(path) {
        return new Promise((resolve, reject) => {
            const image = new Image();
            image.crossOrigin = 'anonymous';
            
            image.onload = () => {
                const texture = _gl.createTexture();
                _gl.bindTexture(_gl.TEXTURE_2D, texture);
                
                // Flip Y for WebGL
                _gl.pixelStorei(_gl.UNPACK_FLIP_Y_WEBGL, true);
                _gl.pixelStorei(_gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
                
                _gl.texImage2D(_gl.TEXTURE_2D, 0, _gl.RGBA, _gl.RGBA, _gl.UNSIGNED_BYTE, image);
                
                // Set texture parameters
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
    
    /**
     * Load expressions
     */
    async function loadExpressions(modelPath) {
        const expressionDefs = _modelSetting.FileReferences.Expressions || [];
        _expressions.clear();
        
        for (const expDef of expressionDefs) {
            try {
                const expPath = modelPath + 'exp/' + expDef.File;
                const response = await fetch(expPath);
                if (response.ok) {
                    const expData = await response.json();
                    _expressions.set(expDef.Name, expData);
                    console.log('Loaded expression:', expDef.Name);
                }
            } catch (error) {
                console.warn(`Failed to load expression ${expDef.Name}:`, error);
            }
        }
        
        return _expressions;
    }
    
    /**
     * Load physics settings
     */
    async function loadPhysics(modelPath) {
        const physicsFile = _modelSetting.FileReferences.Physics;
        if (!physicsFile) return null;
        
        try {
            const physicsPath = modelPath + physicsFile;
            const response = await fetch(physicsPath);
            if (response.ok) {
                const physicsData = await response.json();
                console.log('Physics loaded');
                return physicsData;
            }
        } catch (error) {
            console.warn('Failed to load physics:', error);
        }
        return null;
    }
    
    /**
     * Update projection matrix based on canvas size
     */
    function updateProjectionMatrix() {
        const aspect = _canvas.width / _canvas.height;
        
        // Create orthographic projection
        _projectionMatrix = new Float32Array([
            _scale / aspect, 0, 0, 0,
            0, _scale, 0, 0,
            0, 0, 1, 0,
            _position.x, _position.y, 0, 1
        ]);
    }
    
    /**
     * Set model parameter by ID
     */
    function setParameter(paramId, value) {
        if (!_model) return;
        
        const index = findParameterIndex(paramId);
        if (index >= 0) {
            const minVal = _model.parameters.minimumValues[index];
            const maxVal = _model.parameters.maximumValues[index];
            const clampedValue = Math.max(minVal, Math.min(maxVal, value));
            _model.parameters.values[index] = clampedValue;
        }
    }
    
    /**
     * Get model parameter by ID
     */
    function getParameter(paramId) {
        if (!_model) return 0;
        
        const index = findParameterIndex(paramId);
        if (index >= 0) {
            return _model.parameters.values[index];
        }
        return 0;
    }
    
    /**
     * Find parameter index by ID
     */
    function findParameterIndex(paramId) {
        if (!_model) return -1;
        
        for (let i = 0; i < _model.parameters.count; i++) {
            if (_model.parameters.ids[i] === paramId) {
                return i;
            }
        }
        return -1;
    }
    
    /**
     * Update eye blink animation
     */
    function updateEyeBlink(deltaTime) {
        const now = performance.now() / 1000;
        
        if (!_eyeBlinkState.isBlinking) {
            // Check if it's time to blink
            if (now >= _eyeBlinkState.nextBlinkTime) {
                _eyeBlinkState.isBlinking = true;
                _eyeBlinkState.blinkProgress = 0;
            }
        }
        
        if (_eyeBlinkState.isBlinking) {
            // Blink animation (close then open)
            _eyeBlinkState.blinkProgress += deltaTime * 8; // Speed of blink
            
            if (_eyeBlinkState.blinkProgress < 0.5) {
                // Closing
                const t = _eyeBlinkState.blinkProgress * 2;
                _eyeBlinkState.leftOpen = 1 - t;
                _eyeBlinkState.rightOpen = 1 - t;
            } else if (_eyeBlinkState.blinkProgress < 1.0) {
                // Opening
                const t = (_eyeBlinkState.blinkProgress - 0.5) * 2;
                _eyeBlinkState.leftOpen = t;
                _eyeBlinkState.rightOpen = t;
            } else {
                // Blink complete
                _eyeBlinkState.isBlinking = false;
                _eyeBlinkState.leftOpen = 1;
                _eyeBlinkState.rightOpen = 1;
                // Schedule next blink (2-6 seconds)
                _eyeBlinkState.nextBlinkTime = now + 2 + Math.random() * 4;
            }
        }
        
        // Apply to model
        setParameter(PARAM_IDS.eyeLOpen, _eyeBlinkState.leftOpen);
        setParameter(PARAM_IDS.eyeROpen, _eyeBlinkState.rightOpen);
    }
    
    /**
     * Update breathing animation
     */
    function updateBreathing(deltaTime) {
        const time = performance.now() / 1000;
        _breathValue = (Math.sin(time * 2) + 1) * 0.5; // 0-1 oscillation
        
        setParameter(PARAM_IDS.breath, _breathValue);
        
        // Subtle body movement with breathing
        const bodyAngle = Math.sin(time * 2) * 2;
        setParameter(PARAM_IDS.bodyAngleY, bodyAngle);
    }
    
    /**
     * Update lip sync
     */
    function updateLipSync(deltaTime) {
        // Smooth transition to target value
        _lipSyncValue += (_targetLipSyncValue - _lipSyncValue) * _lipSyncSmoothing;
        
        // Apply to mouth parameter
        setParameter(PARAM_IDS.mouthOpenY, _lipSyncValue);
        
        // Update debug display
        if (_debugMode) {
            const lipEl = document.getElementById('lip-value');
            if (lipEl) lipEl.textContent = _lipSyncValue.toFixed(2);
        }
    }
    
    /**
     * Update head tracking (follow mouse or set position)
     */
    function updateHeadTracking(deltaTime) {
        const smoothing = 0.1;
        
        _headAngle.x += (_targetHeadAngle.x - _headAngle.x) * smoothing;
        _headAngle.y += (_targetHeadAngle.y - _headAngle.y) * smoothing;
        _headAngle.z += (_targetHeadAngle.z - _headAngle.z) * smoothing;
        
        setParameter(PARAM_IDS.angleX, _headAngle.x);
        setParameter(PARAM_IDS.angleY, _headAngle.y);
        setParameter(PARAM_IDS.angleZ, _headAngle.z);
        
        // Eye follow (scaled down from head)
        setParameter(PARAM_IDS.eyeBallX, _headAngle.x / 30);
        setParameter(PARAM_IDS.eyeBallY, _headAngle.y / 30);
    }
    
    /**
     * Main update loop
     */
    function update(timestamp) {
        // Calculate delta time
        _deltaTime = (timestamp - _lastFrameTime) / 1000;
        _lastFrameTime = timestamp;
        
        // Clamp delta time to avoid large jumps
        if (_deltaTime > 0.1) _deltaTime = 0.1;
        
        // Update FPS counter
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
            // Update animations
            updateEyeBlink(_deltaTime);
            updateBreathing(_deltaTime);
            updateLipSync(_deltaTime);
            updateHeadTracking(_deltaTime);
            
            // Update model
            _model.update();
            
            // Render
            render();
        }
        
        // Continue loop
        requestAnimationFrame(update);
    }
    
    /**
     * WebGL Shader programs
     */
    let _shaderProgram = null;
    let _shaderLocations = {};
    
    /**
     * Initialize shaders for rendering
     */
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
                gl_FragColor = texColor * u_baseColor * u_opacity;
            }
        `;
        
        // Compile shaders
        const vertexShader = compileShader(_gl.VERTEX_SHADER, vertexShaderSource);
        const fragmentShader = compileShader(_gl.FRAGMENT_SHADER, fragmentShaderSource);
        
        // Create program
        _shaderProgram = _gl.createProgram();
        _gl.attachShader(_shaderProgram, vertexShader);
        _gl.attachShader(_shaderProgram, fragmentShader);
        _gl.linkProgram(_shaderProgram);
        
        if (!_gl.getProgramParameter(_shaderProgram, _gl.LINK_STATUS)) {
            throw new Error('Shader program link failed: ' + _gl.getProgramInfoLog(_shaderProgram));
        }
        
        // Get locations
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
    
    /**
     * Compile a shader
     */
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
    
    /**
     * Render the model
     */
    function render() {
        // Clear canvas with transparency
        _gl.clear(_gl.COLOR_BUFFER_BIT);
        
        if (!_model || !_shaderProgram || _textures.length === 0) return;
        
        _gl.useProgram(_shaderProgram);
        
        // Get drawable data
        const drawableCount = _model.drawables.count;
        const renderOrders = _model.drawables.renderOrders;
        const opacities = _model.drawables.opacities;
        const textureIndices = _model.drawables.textureIndices;
        const vertexPositions = _model.drawables.vertexPositions;
        const vertexUvs = _model.drawables.vertexUvs;
        const indices = _model.drawables.indices;
        const dynamicFlags = _model.drawables.dynamicFlags;
        
        // Create sorted index array by render order
        const sortedIndices = [];
        for (let i = 0; i < drawableCount; i++) {
            sortedIndices.push(i);
        }
        sortedIndices.sort((a, b) => renderOrders[a] - renderOrders[b]);
        
        // Draw each drawable in order
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
    
    /**
     * Set up mouse tracking for head movement
     */
    function setupMouseTracking() {
        document.addEventListener('mousemove', (e) => {
            const rect = _canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width * 2 - 1;
            const y = -((e.clientY - rect.top) / rect.height * 2 - 1);
            
            // Map to angle range (-30 to 30 degrees)
            _targetHeadAngle.x = x * 30;
            _targetHeadAngle.y = y * 30;
        });
    }
    
    /**
     * Set up audio analysis for lip sync
     */
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
    
    /**
     * Analyze audio and update lip sync
     */
    function analyzeAudioForLipSync() {
        if (!_analyser || !_isLipSyncActive) return;
        
        const dataArray = new Uint8Array(_analyser.frequencyBinCount);
        _analyser.getByteFrequencyData(dataArray);
        
        // Focus on voice frequency range (roughly 100-1000 Hz)
        let sum = 0;
        const startBin = 2;  // ~86 Hz at 44100 sample rate
        const endBin = 24;   // ~1033 Hz
        
        for (let i = startBin; i < endBin && i < dataArray.length; i++) {
            sum += dataArray[i];
        }
        
        const average = sum / (endBin - startBin);
        
        // Normalize and apply threshold
        let normalizedValue = average / 128;  // 0-1 range
        normalizedValue = Math.max(0, normalizedValue - 0.1) * 1.5;  // Apply threshold
        normalizedValue = Math.min(1, normalizedValue);
        
        _targetLipSyncValue = normalizedValue;
        
        // Continue analysis
        if (_isLipSyncActive) {
            requestAnimationFrame(analyzeAudioForLipSync);
        }
    }
    
    // Public API
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
                // Initialize WebGL
                initWebGL(config.canvasId);
                
                // Initialize shaders
                initShaders();
                
                // Load model
                await loadModelSetting(config.modelPath, config.modelName);
                await loadMoc(config.modelPath);
                await loadTextures(config.modelPath);
                await loadExpressions(config.modelPath);
                await loadPhysics(config.modelPath);
                
                // Update UI
                if (_debugMode) {
                    const modelNameEl = document.getElementById('model-name');
                    if (modelNameEl) modelNameEl.textContent = config.modelName;
                }
                
                // Set up projection matrix
                updateProjectionMatrix();
                
                // Set up mouse tracking
                setupMouseTracking();
                
                // Set up audio analysis
                await setupAudioAnalysis();
                
                // Initialize eye blink timing
                _eyeBlinkState.nextBlinkTime = performance.now() / 1000 + 2;
                
                // Start animation loop
                _lastFrameTime = performance.now();
                requestAnimationFrame(update);
                
                _initialized = true;
                console.log('Live2D Manager initialized successfully');
                
            } catch (error) {
                console.error('Failed to initialize Live2D:', error);
                throw error;
            }
        },
        
        /**
         * Set lip sync value directly (0.0 - 1.0)
         */
        setLipSync(value) {
            _targetLipSyncValue = Math.max(0, Math.min(1, value));
        },
        
        /**
         * Start lip sync from audio analysis
         */
        async startAudioLipSync() {
            if (!_audioContext || !_analyser) return;
            
            try {
                // Get microphone or audio element
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
        
        /**
         * Connect an audio element for lip sync
         */
        connectAudioElement(audioElement) {
            if (!_audioContext || !_analyser) return;
            
            try {
                _audioSource = _audioContext.createMediaElementSource(audioElement);
                _audioSource.connect(_analyser);
                _audioSource.connect(_audioContext.destination);
                
                _isLipSyncActive = true;
                analyzeAudioForLipSync();
                
                console.log('Audio element connected for lip sync');
            } catch (error) {
                console.warn('Failed to connect audio element:', error);
            }
        },
        
        /**
         * Stop lip sync
         */
        stopLipSync() {
            _isLipSyncActive = false;
            _targetLipSyncValue = 0;
            
            if (_audioSource) {
                _audioSource.disconnect();
                _audioSource = null;
            }
        },
        
        /**
         * Set expression
         */
        setExpression(name) {
            // Map common emotion names to model expressions
            const expressionMap = {
                // Model expression names (Chinese)
                '捂脸': '捂脸',     // Face cover
                '比耶': '比耶',     // Peace sign
                '照相': '照相',     // Photo
                '脸红': '脸红',     // Blush
                '黑脸': '黑脸',     // Dark face
                '哭': '哭',         // Cry
                '流汗': '流汗',     // Sweat
                '星星': '星星',     // Stars
                // Common names mapping
                'happy': '星星',
                'excited': '比耶',
                'shy': '脸红',
                'blush': '脸红',
                'sad': '哭',
                'cry': '哭',
                'embarrassed': '流汗',
                'sweat': '流汗',
                'angry': '黑脸',
                'photo': '照相',
                'peace': '比耶',
                'cover': '捂脸'
            };
            
            const mappedName = expressionMap[name] || name;
            const expression = _expressions.get(mappedName);
            
            if (expression) {
                _currentExpression = mappedName;
                
                // Apply expression parameters
                for (const param of expression.Parameters) {
                    setParameter(param.Id, param.Value);
                }
                
                if (_debugMode) {
                    const expEl = document.getElementById('expression-name');
                    if (expEl) expEl.textContent = mappedName;
                }
                
                console.log('Expression set:', mappedName);
            } else {
                console.warn('Expression not found:', name);
            }
        },
        
        /**
         * Get available expression names
         */
        getExpressionNames() {
            return Array.from(_expressions.keys());
        },
        
        /**
         * Set parameter directly
         */
        setParameter(paramId, value) {
            setParameter(paramId, value);
        },
        
        /**
         * Get parameter value
         */
        getParameter(paramId) {
            return getParameter(paramId);
        },
        
        /**
         * Play motion (stub - requires motion loader)
         */
        playMotion(group, index) {
            console.log('Play motion:', group, index);
            // TODO: Implement motion playback
        },
        
        /**
         * Get motion groups (stub)
         */
        getMotionGroups() {
            // Return empty for now
            return [];
        },
        
        /**
         * Set head angle
         */
        setHeadAngle(x, y, z = 0) {
            _targetHeadAngle.x = x;
            _targetHeadAngle.y = y;
            _targetHeadAngle.z = z;
        },
        
        /**
         * Set model position
         */
        setPosition(x, y) {
            _position.x = x;
            _position.y = y;
            updateProjectionMatrix();
        },
        
        /**
         * Set model scale
         */
        setScale(scale) {
            _scale = scale;
            updateProjectionMatrix();
        },
        
        /**
         * Toggle debug mode
         */
        toggleDebug() {
            _debugMode = !_debugMode;
            return _debugMode;
        },
        
        /**
         * Check if initialized
         */
        isInitialized() {
            return _initialized;
        },
        
        /**
         * Get current FPS
         */
        getFPS() {
            return _fps;
        }
    };
})();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Live2DManager;
}
