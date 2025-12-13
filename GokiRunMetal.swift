import Cocoa
import Metal
import QuartzCore
import simd

// MARK: - Goki State (shared with Metal)

struct GokiState {
  var position: SIMD2<Float>
  var velocity: SIMD2<Float>
  var targetAngle: Float
  var currentAngle: Float
  var state: Int32  // 0=IDLE, 1=RANDOM_WALK, 2=WALL_FOLLOW, 3=ESCAPE, 4=STOP, 5=DEAD, 6=REVIVING
  var stateTimer: Float  // used for escape duration and stop duration
  var randomSeed: UInt32
  var wasUnderWindow: Int32 = 0  // 前フレームでウィンドウ下にいたか
  var deathTimer: Float = 0  // 死亡/復活タイマー
  var _padding3: Float = 0  // アライメント調整
}

// MARK: - Window Info (for hiding under windows)

struct WindowInfo {
  var rect: SIMD4<Float>  // (minX, minY, maxX, maxY)
}

struct SimulationParams {
  var mousePosition: SIMD2<Float>
  var screenMin: SIMD2<Float>  // visibleFrame origin (excludes Dock)
  var screenMax: SIMD2<Float>  // visibleFrame max (excludes menu bar)
  var deltaTime: Float
  var threatRadius: Float
  var wallRadius: Float
  var maxSpeed: Float
  var escapeAcceleration: Float
  var friction: Float
  var rotationSpeed: Float
  var gokiCount: Int32
  var frameCount: UInt32
  var windowCount: Int32
  var _padding2: Int32 = 0  // アライメント調整
}

// MARK: - Metal Shader Source

let shaderSource = """
  #include <metal_stdlib>
  using namespace metal;

  struct GokiState {
      float2 position;
      float2 velocity;
      float targetAngle;
      float currentAngle;
      int state;          // 0=IDLE, 1=RANDOM_WALK, 2=WALL_FOLLOW, 3=ESCAPE, 4=STOP, 5=DEAD, 6=REVIVING
      float stateTimer;
      uint randomSeed;
      int wasUnderWindow;
      float deathTimer;   // 死亡/復活タイマー
      float _padding3;
  };

  struct SimulationParams {
      float2 mousePosition;
      float2 screenMin;    // visibleFrame origin
      float2 screenMax;    // visibleFrame max
      float deltaTime;
      float threatRadius;
      float wallRadius;
      float maxSpeed;
      float escapeAcceleration;
      float friction;
      float rotationSpeed;
      int gokiCount;
      uint frameCount;
      int windowCount;
      int _padding2;
  };

  struct WindowInfo {
      float4 rect;  // (minX, minY, maxX, maxY)
  };

  // PCG Random
  uint pcg_hash(uint input) {
      uint state = input * 747796405u + 2891336453u;
      uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
      return (word >> 22u) ^ word;
  }

  float random_float(thread uint* seed) {
      *seed = pcg_hash(*seed);
      return float(*seed) / float(0xFFFFFFFFu);
  }

  // Preferred escape angles (90, 120, 150, 180 degrees in radians)
  constant float escapeAngles[4] = {1.5708, 2.0944, 2.618, 3.1416};

  // Check if position is under any window
  bool isUnderWindow(float2 pos, constant WindowInfo* windows, int windowCount) {
      for (int i = 0; i < windowCount; i++) {
          float4 r = windows[i].rect;
          if (pos.x >= r.x && pos.x <= r.z && pos.y >= r.y && pos.y <= r.w) {
              return true;
          }
      }
      return false;
  }

  kernel void updateGoki(
      device GokiState* gokis [[buffer(0)]],
      constant SimulationParams& params [[buffer(1)]],
      constant WindowInfo* windows [[buffer(2)]],
      uint id [[thread_position_in_grid]]
  ) {
      if (id >= uint(params.gokiCount)) return;

      device GokiState& g = gokis[id];
      uint seed = g.randomSeed ^ (params.frameCount * 1000u + id);

      // DEAD/REVIVINGの場合は何もしない（シェーダー内で処理）
      if (g.state == 5 || g.state == 6) {
          // タイマー処理は下で行う
      }

      // Check window exposure (was under window but now exposed)
      bool currentlyUnderWindow = isUnderWindow(g.position, windows, params.windowCount);
      if (g.wasUnderWindow && !currentlyUnderWindow && g.state != 3 && g.state != 5 && g.state != 6) {
          // Exposed! Freeze for 0.5 seconds then escape
          g.state = 4; // STOP (freeze)
          g.stateTimer = 0.5; // Freeze duration
          g.wasUnderWindow = 0;
      } else {
          g.wasUnderWindow = currentlyUnderWindow ? 1 : 0;
      }

      float2 toMouse = params.mousePosition - g.position;
      float distToMouse = length(toMouse);

      // Two-stage threat response (based on Camhi & Nolen 1981)
      // - Weak stimulus (far) -> pause
      // - Strong stimulus (near) -> escape
      float nearThreshold = params.threatRadius * 0.5;   // Strong stimulus zone
      float farThreshold = params.threatRadius;          // Weak stimulus zone

      if (g.state != 3 && g.state != 5 && g.state != 6) {
          if (distToMouse < nearThreshold) {
              // Strong stimulus: ESCAPE
              g.state = 3;
              g.stateTimer = 0.5 + random_float(&seed) * 0.5;

              // Select preferred escape trajectory (90, 120, 150, 180 degrees)
              int etIndex = int(random_float(&seed) * 4.0) % 4;
              float baseAngle = escapeAngles[etIndex];

              // Add Gaussian-like noise (~15 degrees)
              float noise = (random_float(&seed) - 0.5) * 0.5;

              // Calculate escape direction (away from threat)
              float threatAngle = atan2(toMouse.y, toMouse.x);

              // 90% away, 10% towards
              if (random_float(&seed) < 0.9) {
                  g.targetAngle = threatAngle + 3.1416 + baseAngle - 1.5708 + noise;
              } else {
                  // Towards response (small turn <30 degrees)
                  g.targetAngle = threatAngle + (random_float(&seed) - 0.5) * 0.5;
              }
          }
          else if (distToMouse < farThreshold && g.state != 4) {
              // Weak stimulus: PAUSE (stop walking)
              g.state = 4;
              g.stateTimer = 0.3 + random_float(&seed) * 0.5;
          }
      }

      // Update based on state
      if (g.state == 3) {
          // ESCAPE: rotate then accelerate
          float angleDiff = g.targetAngle - g.currentAngle;

          // Normalize angle difference
          while (angleDiff > 3.1416) angleDiff -= 6.2832;
          while (angleDiff < -3.1416) angleDiff += 6.2832;

          if (abs(angleDiff) > 0.1) {
              g.currentAngle += sign(angleDiff) * params.rotationSpeed * params.deltaTime;
          }

          // Accelerate in current direction
          float2 dir = float2(cos(g.currentAngle), sin(g.currentAngle));
          g.velocity += dir * params.escapeAcceleration;

          g.stateTimer -= params.deltaTime;
          if (g.stateTimer <= 0 || distToMouse > params.threatRadius * 2.0) {
              g.state = 1; // RANDOM_WALK
          }
      }
      else if (g.state == 4) {
          // STOP: stay still, occasionally twitch antennae (via small angle change)
          g.velocity *= 0.8; // Slow down quickly

          if (random_float(&seed) < 0.01) {
              g.currentAngle += (random_float(&seed) - 0.5) * 0.2; // Small twitch
          }

          g.stateTimer -= params.deltaTime;
          if (g.stateTimer <= 0) {
              // If exposed (not under window and not near wall), run away!
              bool nearWall = g.position.x < params.screenMin.x + params.wallRadius ||
                              g.position.x > params.screenMax.x - params.wallRadius ||
                              g.position.y < params.screenMin.y + params.wallRadius ||
                              g.position.y > params.screenMax.y - params.wallRadius;
              if (!currentlyUnderWindow && !nearWall) {
                  // Exposed in open! Run to nearest wall
                  g.state = 3; // ESCAPE
                  g.stateTimer = 0.3 + random_float(&seed) * 0.3;
                  // Head toward nearest wall
                  float2 center = (params.screenMin + params.screenMax) * 0.5;
                  float2 fromCenter = g.position - center;
                  g.targetAngle = atan2(fromCenter.y, fromCenter.x);
              } else {
                  g.state = 1; // Resume walking
              }
          }
      }
      else if (g.state == 1) {
          // RANDOM_WALK: G is exploring, but nervous in the open

          // More frequent direction changes (G is cautious in open space)
          if (random_float(&seed) < 0.03) {
              g.targetAngle = random_float(&seed) * 6.2832;
          }

          // Sometimes feel unsafe and stop to check surroundings
          if (random_float(&seed) < 0.002) {
              g.state = 4;
              g.stateTimer = 0.5 + random_float(&seed) * 1.5; // Short pause
          }

          // G wants to find a wall (safety!) - bias toward nearest wall
          float2 screenCenter = (params.screenMin + params.screenMax) * 0.5;
          float2 toCenter = screenCenter - g.position;
          float distToCenter = length(toCenter);
          float maxDist = length(params.screenMax - params.screenMin) * 0.5;

          // The further from center, the more likely to head toward a wall
          // But sometimes venture toward center for food
          if (random_float(&seed) < 0.01) {
              if (distToCenter < maxDist * 0.3) {
                  // Near center: 70% chance to head toward wall, 30% continue exploring
                  if (random_float(&seed) < 0.7) {
                      // Pick a random wall direction
                      float wallAngle = float(int(random_float(&seed) * 4.0)) * 1.5708;
                      g.targetAngle = wallAngle;
                  }
              }
          }

          float angleDiff = g.targetAngle - g.currentAngle;
          while (angleDiff > 3.1416) angleDiff -= 6.2832;
          while (angleDiff < -3.1416) angleDiff += 6.2832;

          g.currentAngle += sign(angleDiff) * min(abs(angleDiff), params.rotationSpeed * params.deltaTime * 0.3);

          float2 dir = float2(cos(g.currentAngle), sin(g.currentAngle));
          g.velocity += dir * 0.5;

          // Check wall proximity - G found safety!
          if (g.position.x < params.screenMin.x + params.wallRadius || g.position.x > params.screenMax.x - params.wallRadius ||
              g.position.y < params.screenMin.y + params.wallRadius || g.position.y > params.screenMax.y - params.wallRadius) {
              g.state = 2; // WALL_FOLLOW
              g.stateTimer = 3.0 + random_float(&seed) * 7.0; // How long to stay on wall
          }

          // Check if under window - even safer than wall!
          bool underWindow = isUnderWindow(g.position, windows, params.windowCount);
          if (underWindow) {
              // Under window is super safe - slow down and maybe stop
              g.velocity *= 0.85;
              if (random_float(&seed) < 0.03) {
                  g.state = 4; // STOP
                  g.stateTimer = 2.0 + random_float(&seed) * 5.0; // Long rest
              }
          }
      }
      else if (g.state == 2) {
          // WALL_FOLLOW: G feels safe here, but will eventually leave
          float2 wallNormal = float2(0, 0);
          bool inCorner = false;

          if (g.position.x < params.screenMin.x + params.wallRadius) wallNormal.x = 1;
          if (g.position.x > params.screenMax.x - params.wallRadius) wallNormal.x = -1;
          if (g.position.y < params.screenMin.y + params.wallRadius) wallNormal.y = 1;
          if (g.position.y > params.screenMax.y - params.wallRadius) wallNormal.y = -1;

          // Detect corner (two walls) - extra safe!
          if (abs(wallNormal.x) > 0 && abs(wallNormal.y) > 0) {
              inCorner = true;
          }

          // Decrease wall timer
          g.stateTimer -= params.deltaTime;

          if (length(wallNormal) > 0) {
              // Stop in corners (feels very safe)
              float stopChance = inCorner ? 0.01 : 0.003;
              if (random_float(&seed) < stopChance) {
                  g.state = 4; // STOP
                  g.stateTimer = inCorner ? (3.0 + random_float(&seed) * 5.0) : (1.0 + random_float(&seed) * 2.0);
              }
              // Time to leave wall and explore? (looking for food)
              else if (g.stateTimer <= 0 && random_float(&seed) < 0.02) {
                  g.state = 1; // Back to random walk
                  // Turn away from wall
                  g.targetAngle = atan2(wallNormal.y, wallNormal.x) + (random_float(&seed) - 0.5) * 1.0;
                  g.velocity += wallNormal * 2.0; // Push away from wall
              }
              else {
                  // Move along wall
                  float2 tangent = float2(-wallNormal.y, wallNormal.x);
                  if (dot(g.velocity, tangent) < 0) tangent = -tangent;

                  // Occasionally reverse direction on wall
                  if (random_float(&seed) < 0.005) {
                      tangent = -tangent;
                  }

                  g.targetAngle = atan2(tangent.y, tangent.x);
                  g.velocity += tangent * 0.3;
              }
          } else {
              g.state = 1; // Lost the wall, back to random walk
          }
      }
      else if (g.state == 5) {
          // DEAD: 潰れた状態、タイマー減少
          g.velocity = float2(0, 0);
          g.deathTimer -= params.deltaTime;
          if (g.deathTimer <= 0) {
              g.state = 6;  // REVIVING
              g.deathTimer = 1.0;  // 復活エフェクト時間
          }
      }
      else if (g.state == 6) {
          // REVIVING: 復活中
          g.velocity = float2(0, 0);
          g.deathTimer -= params.deltaTime;
          if (g.deathTimer <= 0) {
              g.state = 1;  // RANDOM_WALK
              g.deathTimer = 0;
          }
      }
      else {
          // IDLE: start walking
          g.state = 1;
      }

      // DEAD/REVIVINGの場合は移動しない
      if (g.state == 5 || g.state == 6) {
          g.randomSeed = seed;
          return;
      }

      // Clamp speed
      float speed = length(g.velocity);
      if (speed > params.maxSpeed) {
          g.velocity = g.velocity / speed * params.maxSpeed;
      }

      // Apply velocity
      g.position += g.velocity * params.deltaTime * 60.0;

      // Apply friction
      g.velocity *= params.friction;

      // Bounce off walls (constrained to visible area)
      if (g.position.x < params.screenMin.x) { g.position.x = params.screenMin.x; g.velocity.x *= -0.5; }
      if (g.position.x > params.screenMax.x) { g.position.x = params.screenMax.x; g.velocity.x *= -0.5; }
      if (g.position.y < params.screenMin.y) { g.position.y = params.screenMin.y; g.velocity.y *= -0.5; }
      if (g.position.y > params.screenMax.y) { g.position.y = params.screenMax.y; g.velocity.y *= -0.5; }

      g.randomSeed = seed;
  }
  """

// MARK: - Goki Window

class GokiWindow: NSWindow {
  static let gokiSize: CGFloat = 60  // 復活エフェクト用に大きく

  init() {
    let size = GokiWindow.gokiSize
    super.init(
      contentRect: NSRect(x: 0, y: 0, width: size, height: size),
      styleMask: .borderless,
      backing: .buffered,
      defer: false
    )
    isOpaque = false
    backgroundColor = .clear
    hasShadow = false
    level = NSWindow.Level(rawValue: -1)
    collectionBehavior = [.canJoinAllSpaces, .stationary]
    ignoresMouseEvents = true  // クリックはグローバル監視で処理
    contentView = GokiView(frame: NSRect(x: 0, y: 0, width: size, height: size))
  }
}

class GokiView: NSView {
  var angle: CGFloat = 0
  var squashScale: CGFloat = 1.0  // 1.0=通常, 0.1=潰れた
  var reviveGlow: CGFloat = 0.0   // 0.0=なし, 1.0=最大輝度

  override func draw(_ dirtyRect: NSRect) {
    guard let ctx = NSGraphicsContext.current?.cgContext else { return }

    let scale: CGFloat = 1.0  // Gは固定サイズ（元の24px基準）

    ctx.saveGState()
    ctx.translateBy(x: bounds.midX, y: bounds.midY)
    ctx.rotate(by: angle)
    ctx.scaleBy(x: scale, y: scale)

    // 潰れアニメーション: Y方向にスケール
    ctx.scaleBy(x: 1.0 + (1.0 - squashScale) * 0.3, y: squashScale)  // 潰れると横に広がる

    let gColor = NSColor(red: 0.12, green: 0.06, blue: 0.03, alpha: 1).cgColor

    // Body - ぷっくり楕円
    ctx.setFillColor(gColor)
    ctx.fillEllipse(in: CGRect(x: -10, y: -6, width: 18, height: 12))

    // Antennae - 楕円の前端から生える触覚
    ctx.setStrokeColor(gColor)
    ctx.setLineWidth(1.0)
    ctx.setLineCap(.round)
    ctx.beginPath()
    ctx.move(to: CGPoint(x: 7, y: 2))
    ctx.addQuadCurve(to: CGPoint(x: 20, y: 9), control: CGPoint(x: 14, y: 3))
    ctx.move(to: CGPoint(x: 7, y: -2))
    ctx.addQuadCurve(to: CGPoint(x: 20, y: -9), control: CGPoint(x: 14, y: -3))
    ctx.strokePath()

    ctx.restoreGState()

    // 復活エフェクト: 謎の光のリング
    if reviveGlow > 0 {
      ctx.saveGState()
      ctx.translateBy(x: bounds.midX, y: bounds.midY)

      // 外側のリング（大きく収縮）
      let outerRingSize = 25 + (1.0 - reviveGlow) * 30
      ctx.setStrokeColor(NSColor(red: 0.4, green: 1.0, blue: 0.3, alpha: reviveGlow * 0.9).cgColor)
      ctx.setLineWidth(3.0)
      ctx.strokeEllipse(in: CGRect(x: -outerRingSize, y: -outerRingSize, width: outerRingSize * 2, height: outerRingSize * 2))

      // 中間のリング（逆方向に拡大）
      let midRingSize = 18 + reviveGlow * 12
      ctx.setStrokeColor(NSColor(red: 0.6, green: 1.0, blue: 0.5, alpha: reviveGlow * 0.7).cgColor)
      ctx.setLineWidth(2.0)
      ctx.strokeEllipse(in: CGRect(x: -midRingSize, y: -midRingSize, width: midRingSize * 2, height: midRingSize * 2))

      // 内側の光（グロー）
      let innerGlowSize: CGFloat = 20
      ctx.setFillColor(NSColor(red: 0.8, green: 1.0, blue: 0.6, alpha: reviveGlow * 0.4).cgColor)
      ctx.fillEllipse(in: CGRect(x: -innerGlowSize, y: -innerGlowSize, width: innerGlowSize * 2, height: innerGlowSize * 2))

      // 中心のコア（明るい）
      let coreSize: CGFloat = 8
      ctx.setFillColor(NSColor(red: 1.0, green: 1.0, blue: 0.9, alpha: reviveGlow * 0.8).cgColor)
      ctx.fillEllipse(in: CGRect(x: -coreSize, y: -coreSize, width: coreSize * 2, height: coreSize * 2))

      // 放射状の光線（4本）
      ctx.setStrokeColor(NSColor(red: 0.7, green: 1.0, blue: 0.5, alpha: reviveGlow * 0.6).cgColor)
      ctx.setLineWidth(1.5)
      let rayLength = 20 + (1.0 - reviveGlow) * 10
      for i in 0..<4 {
        let angle = CGFloat(i) * .pi / 2 + reviveGlow * .pi / 4
        ctx.move(to: CGPoint(x: cos(angle) * 10, y: sin(angle) * 10))
        ctx.addLine(to: CGPoint(x: cos(angle) * rayLength, y: sin(angle) * rayLength))
      }
      ctx.strokePath()

      ctx.restoreGState()
    }
  }
}

// MARK: - App Delegate

class AppDelegate: NSObject, NSApplicationDelegate {
  var device: MTLDevice!
  var commandQueue: MTLCommandQueue!
  var computePipeline: MTLComputePipelineState!
  var gokiBuffer: MTLBuffer!
  var paramsBuffer: MTLBuffer!
  var windowBuffer: MTLBuffer!

  var gokiWindows: [GokiWindow] = []
  var gokiStates: [GokiState] = []
  var displayLink: CADisplayLink?

  // Window detection
  var windowCount: Int = 0
  var lastWindowUpdate: CFAbsoluteTime = 0
  let maxWindows = 32

  let gokiCount: Int

  init(count: Int) {
    self.gokiCount = count
    super.init()
  }

  func applicationDidFinishLaunching(_ notification: Notification) {
    guard let screen = NSScreen.main else { return }
    let visibleFrame = screen.visibleFrame  // Excludes menu bar and Dock

    // Setup Metal
    guard setupMetal() else {
      print("Metal setup failed")
      return
    }

    // Initialize Goki states within visible area
    for i in 0..<gokiCount {
      let state = GokiState(
        position: SIMD2<Float>(
          Float.random(in: Float(visibleFrame.minX + 50)...Float(visibleFrame.maxX - 50)),
          Float.random(in: Float(visibleFrame.minY + 50)...Float(visibleFrame.maxY - 50))
        ),
        velocity: SIMD2<Float>(0, 0),
        targetAngle: Float.random(in: 0...Float.pi * 2),
        currentAngle: Float.random(in: 0...Float.pi * 2),
        state: 1,
        stateTimer: 0,
        randomSeed: UInt32(i * 12345 + 67890)
      )
      gokiStates.append(state)

      let window = GokiWindow()
      window.setFrameOrigin(NSPoint(x: CGFloat(state.position.x), y: CGFloat(state.position.y)))
      window.orderFront(nil)
      gokiWindows.append(window)
    }

    // Create buffer
    gokiBuffer = device.makeBuffer(
      bytes: &gokiStates,
      length: MemoryLayout<GokiState>.stride * gokiCount,
      options: .storageModeShared
    )

    // Start display link
    setupDisplayLink()

    // Mouse tracking
    NSEvent.addGlobalMonitorForEvents(matching: [
      .mouseMoved, .leftMouseDragged, .rightMouseDragged,
    ]) { [weak self] _ in
      self?.updateMousePosition()
    }
    NSEvent.addLocalMonitorForEvents(matching: [.mouseMoved, .leftMouseDragged, .rightMouseDragged])
    { [weak self] event in
      self?.updateMousePosition()
      return event
    }

    // Click monitoring - グローバルにクリック検出して最も近いGを潰す
    NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown]) { [weak self] event in
      self?.handleClick(at: NSEvent.mouseLocation)
    }
    NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown]) { [weak self] event in
      self?.handleClick(at: NSEvent.mouseLocation)
      return event
    }
  }

  func handleClick(at location: NSPoint) {
    let statesPtr = gokiBuffer.contents().bindMemory(to: GokiState.self, capacity: gokiCount)

    var nearestIndex = -1
    var nearestDistance: Float = Float.greatestFiniteMagnitude
    let clickRadius: Float = 20  // クリック判定半径

    for i in 0..<gokiCount {
      let state = statesPtr[i]
      // DEAD/REVIVING中は無視
      if state.state == 5 || state.state == 6 { continue }

      let dx = Float(location.x) - state.position.x
      let dy = Float(location.y) - state.position.y
      let distance = sqrt(dx * dx + dy * dy)

      if distance < clickRadius && distance < nearestDistance {
        nearestDistance = distance
        nearestIndex = i
      }
    }

    if nearestIndex >= 0 {
      squashGoki(at: nearestIndex)
    }
  }

  func setupMetal() -> Bool {
    guard let device = MTLCreateSystemDefaultDevice() else { return false }
    self.device = device

    guard let queue = device.makeCommandQueue() else { return false }
    self.commandQueue = queue

    // Compile shader
    do {
      let library = try device.makeLibrary(source: shaderSource, options: nil)
      guard let function = library.makeFunction(name: "updateGoki") else { return false }
      computePipeline = try device.makeComputePipelineState(function: function)
    } catch {
      print("Shader compilation error: \(error)")
      return false
    }

    // Params buffer
    paramsBuffer = device.makeBuffer(
      length: MemoryLayout<SimulationParams>.stride, options: .storageModeShared)

    // Window buffer for hiding under windows
    windowBuffer = device.makeBuffer(
      length: MemoryLayout<WindowInfo>.stride * maxWindows, options: .storageModeShared)

    return true
  }

  func setupDisplayLink() {
    guard let screen = NSScreen.main else { return }
    displayLink = screen.displayLink(target: self, selector: #selector(displayLinkCallback(_:)))
    displayLink?.add(to: .main, forMode: .common)
  }

  @objc func displayLinkCallback(_ displayLink: CADisplayLink) {
    update()
  }

  var frameCount: UInt32 = 0
  var mousePosition: NSPoint = .zero

  func updateMousePosition() {
    mousePosition = NSEvent.mouseLocation
  }

  func updateWindowRects() {
    let currentTime = CFAbsoluteTimeGetCurrent()
    guard currentTime - lastWindowUpdate > 0.5 else { return }
    lastWindowUpdate = currentTime

    guard let screen = NSScreen.main else { return }
    let screenHeight = screen.frame.height

    let options = CGWindowListOption(arrayLiteral: .optionOnScreenOnly, .excludeDesktopElements)
    guard let windowList = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]]
    else { return }

    var windows: [WindowInfo] = []
    let ownPID = ProcessInfo.processInfo.processIdentifier

    for window in windowList {
      // 自身のウィンドウを除外
      if let pid = window[kCGWindowOwnerPID as String] as? Int32, pid == ownPID { continue }
      // layer <= 0 はGより下なので除外（Gはlevel -1）
      if let layer = window[kCGWindowLayer as String] as? Int, layer <= 0 { continue }

      guard let boundsDict = window[kCGWindowBounds as String] as? [String: CGFloat],
        let bounds = CGRect(dictionaryRepresentation: boundsDict as CFDictionary)
      else { continue }

      // Y座標を反転（CGWindowは上が0、NSWindowは下が0）
      let minY = screenHeight - bounds.maxY
      let maxY = screenHeight - bounds.minY

      windows.append(
        WindowInfo(
          rect: SIMD4<Float>(
            Float(bounds.minX), Float(minY), Float(bounds.maxX), Float(maxY)
          )))

      if windows.count >= maxWindows { break }
    }

    // バッファ更新
    windowCount = windows.count
    if windowCount > 0 {
      memcpy(windowBuffer.contents(), &windows, MemoryLayout<WindowInfo>.stride * windowCount)
    }
  }

  func update() {
    guard let screen = NSScreen.main else { return }
    let visibleFrame = screen.visibleFrame  // Excludes menu bar and Dock

    // Update window rects for hiding
    updateWindowRects()

    // Update params
    var params = SimulationParams(
      mousePosition: SIMD2<Float>(Float(mousePosition.x), Float(mousePosition.y)),
      screenMin: SIMD2<Float>(Float(visibleFrame.minX), Float(visibleFrame.minY)),
      screenMax: SIMD2<Float>(Float(visibleFrame.maxX), Float(visibleFrame.maxY)),
      deltaTime: 1.0 / 60.0,
      threatRadius: 180,
      wallRadius: 30,
      maxSpeed: 14,
      escapeAcceleration: 4,
      friction: 0.94,
      rotationSpeed: 12,
      gokiCount: Int32(gokiCount),
      frameCount: frameCount,
      windowCount: Int32(windowCount)
    )
    memcpy(paramsBuffer.contents(), &params, MemoryLayout<SimulationParams>.stride)

    // Run compute shader
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else { return }

    encoder.setComputePipelineState(computePipeline)
    encoder.setBuffer(gokiBuffer, offset: 0, index: 0)
    encoder.setBuffer(paramsBuffer, offset: 0, index: 1)
    encoder.setBuffer(windowBuffer, offset: 0, index: 2)

    let threadGroupSize = MTLSize(width: min(gokiCount, 64), height: 1, depth: 1)
    let threadGroups = MTLSize(width: (gokiCount + 63) / 64, height: 1, depth: 1)
    encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)

    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // Read back and update windows
    let statesPtr = gokiBuffer.contents().bindMemory(to: GokiState.self, capacity: gokiCount)
    let halfSize = GokiWindow.gokiSize / 2
    for i in 0..<gokiCount {
      let state = statesPtr[i]
      gokiWindows[i].setFrameOrigin(
        NSPoint(x: CGFloat(state.position.x) - halfSize, y: CGFloat(state.position.y) - halfSize))
      if let view = gokiWindows[i].contentView as? GokiView {
        view.angle = CGFloat(state.currentAngle)

        // 潰れ/復活アニメーション
        if state.state == 5 {
          // DEAD: 潰れた
          view.squashScale = max(0.1, CGFloat(state.deathTimer / 2.0))  // 2秒かけて潰れる
          view.reviveGlow = 0
        } else if state.state == 6 {
          // REVIVING: 復活中
          view.squashScale = 1.0 - CGFloat(state.deathTimer)  // 1秒かけて戻る
          view.reviveGlow = CGFloat(state.deathTimer)  // 光が弱まる
        } else {
          view.squashScale = 1.0
          view.reviveGlow = 0
        }

        view.needsDisplay = true
      }
    }

    frameCount += 1
  }

  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
    return false
  }

  func applicationWillTerminate(_ notification: Notification) {
    displayLink?.invalidate()
  }

  func squashGoki(at index: Int) {
    guard index >= 0 && index < gokiCount else { return }
    let statesPtr = gokiBuffer.contents().bindMemory(to: GokiState.self, capacity: gokiCount)
    // DEADでもREVIVINGでもない場合のみ潰す
    if statesPtr[index].state != 5 && statesPtr[index].state != 6 {
      statesPtr[index].state = 5  // DEAD
      statesPtr[index].deathTimer = 2.0  // 2秒後に復活開始
      statesPtr[index].velocity = SIMD2<Float>(0, 0)
    }
  }
}

// MARK: - Argument Parsing

func parseArguments() -> Int {
  let args = CommandLine.arguments
  var count = 10  // default

  for i in 0..<args.count {
    if args[i] == "--count" || args[i] == "-c" {
      if i + 1 < args.count, let value = Int(args[i + 1]) {
        count = max(1, min(value, 1000))  // 1-1000
      }
    }
    if args[i] == "--help" || args[i] == "-h" {
      print("Usage: swift GokiRunMetal.swift [options]")
      print("  -c, --count <n>  Number of G (1-1000, default: 10)")
      exit(0)
    }
  }
  return count
}

// MARK: - Main

@main
struct GokiRunApp {
  static func main() {
    let gokiCountArg = parseArguments()
    let app = NSApplication.shared
    let delegate = AppDelegate(count: gokiCountArg)
    app.delegate = delegate
    app.setActivationPolicy(.accessory)
    app.run()
  }
}
