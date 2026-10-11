import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/jiayiwang/TUM/MA/kuavo_learning_studio";
const SKILL_DIR = "/Users/jiayiwang/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11814/skills/presentations";
const TMP_DIR = path.join(workspaceDir, ".codex-ppt-build/vla-flowchart");
const FINAL_PPTX = path.join(workspaceDir, "output/Kuavo_VLA_执行链路_1.3.3_v2.pptx");
const RUNTIME_PYTHON = "/Users/jiayiwang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const family = "Hiragino Sans GB";

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const s = p.slides.add();
s.background.fill = "#F6F8FB";

function textBox(text, pos, size, color = "#172033", bold = false, align = "left") {
  const sh = s.shapes.add({
    geometry: "textbox", position: pos, fill: "none",
    line: { fill: "none", width: 0 },
  });
  sh.text = text;
  sh.text.style = {
    typeface: family, fontSize: size, color, bold,
    alignment: align, verticalAlignment: "middle",
    autoFit: "shrinkText", marginLeft: 2, marginRight: 2,
    marginTop: 2, marginBottom: 2,
  };
  return sh;
}

function phase(x, w, title, subtitle, fill, accent) {
  const bg = s.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: 112, width: w, height: 526 },
    fill, line: { style: "solid", fill: "#D9E0EA", width: 1 },
    borderRadius: 18,
  });
  const band = s.shapes.add({
    geometry: "roundRect",
    position: { left: x + 12, top: 126, width: w - 24, height: 62 },
    fill: accent, line: { fill: "none", width: 0 }, borderRadius: 14,
  });
  textBox(title, { left: x + 24, top: 132, width: w - 48, height: 28 }, 22, "#FFFFFF", true, "center");
  textBox(subtitle, { left: x + 24, top: 160, width: w - 48, height: 20 }, 13, "#EAF1FF", false, "center");
  return { bg, band };
}

function node(x, y, w, h, title, details, accent, fill = "#FFFFFF") {
  const box = s.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill, line: { style: "solid", fill: accent, width: 2 },
    borderRadius: 12,
  });
  textBox(title, { left: x + 12, top: y + 7, width: w - 24, height: details ? 27 : h - 14 }, details ? 18 : 15, "#172033", true, "center");
  if (details) {
    textBox(details, { left: x + 13, top: y + 36, width: w - 26, height: h - 43 }, 13, "#465269", false, "center");
  }
  return box;
}

function arrow(a, b, fromSide = "bottom", toSide = "top", color = "#607089") {
  const connector = s.shapes.connect(a, b, {
    kind: "straight", fromSide, toSide,
    line: { style: "solid", fill: color, width: 2.2 },
    head: { type: "triangle", width: "med", length: "med" },
  });
  connector.bringToFront();
  return connector;
}

textBox("LingBot-VLA 到 Kuavo v42 真机的执行链路", { left: 54, top: 34, width: 1172, height: 48 }, 34, "#13213A", true);
textBox("kuavo_learning_studio  ×  kuavo-ros-opensource 1.3.3", { left: 56, top: 80, width: 900, height: 24 }, 15, "#66738A");

phase(48, 276, "VLA / Application", "kuavo_learning_studio", "#EDF4FF", "#2864B4");
phase(348, 276, "Kuavo SDK", "kuavo_humanoid_sdk 1.3.3", "#EBF9F5", "#16816B");
phase(648, 330, "ROS Controller / WBC", "humanoidController", "#FFF4E8", "#C66A18");
phase(1002, 230, "Hardware", "真机执行", "#F2F0F8", "#66549A");

const model = node(66, 210, 240, 91, "LingBot-VLA", "模型输出 action 16D\n[L q7, L hand, R q7, R hand]", "#2864B4");
const decode = node(66, 326, 240, 98, "FeatureTransform.unapply()", "反归一化\narm delta + current state\n得到 absolute q，单位 rad", "#2864B4");
const env = node(66, 449, 240, 94, "KuavoBaseRosEnv.exec_action()", "拆出 q14 与 gripper2\nq14 交给 Kuavo SDK", "#2864B4");
const eef = node(90, 568, 192, 50, "_control_eef()\n左右手独立执行链", "", "#7B8DA9", "#F8FAFD");
arrow(model, decode);
arrow(decode, env);
arrow(env, eef, "bottom", "top", "#8A97AA");

const robot = node(366, 210, 240, 98, "KuavoRobot", "control_arm_joint_positions(q14)\n14 DoF 校验与 joint-limit clipping", "#16816B");
const core = node(366, 334, 240, 98, "KuavoRobotCore", "碰撞保护\n切换 ExternalControl", "#16816B");
const rosctl = node(366, 458, 240, 104, "ControlRobotArm", "pub_control_robot_arm_traj()\nrad 转 degree\n构造 sensor_msgs/JointState", "#16816B");
arrow(robot, core);
arrow(core, rosctl);
arrow(env, robot, "right", "left", "#2864B4");

const topic1 = node(672, 204, 282, 67, "/kuavo_arm_traj", "JointState.position[14]，单位 degree", "#C66A18", "#FFF9F2");
const callback = node(672, 292, 282, 88, "humanoidController callback", "degree 转 rad\n写入 arm_joint_trajectory_", "#C66A18");
const target = node(672, 401, 282, 76, "控制循环", "滤波并写入 optimizedState2WBC_mrt_", "#C66A18");
const wbc = node(672, 498, 282, 73, "WBC update()", "生成 output_pos_ / output_vel_ / output_tau_", "#C66A18");
const topic2 = node(696, 584, 234, 54, "/joint_cmd\nkuavo_msgs/jointCmd", "", "#C66A18", "#FFF9F2");
arrow(topic1, callback);
arrow(callback, target);
arrow(target, wbc);
arrow(wbc, topic2);
arrow(rosctl, topic1, "right", "left", "#16816B");

const nodelet = node(1020, 220, 194, 91, "HardwareNodelet", "HardwareNodeletCXX\n订阅 /joint_cmd", "#66549A");
const plant = node(1020, 350, 194, 86, "hardware_plant", "HardwarePlant\n硬件抽象与命令下发", "#66549A");
const motors = node(1020, 482, 194, 97, "真实 Motors", "PA100 / EtherCAT\nRuiwo actuator", "#66549A", "#FBFAFE");
arrow(nodelet, plant, "bottom", "top", "#66549A");
arrow(plant, motors, "bottom", "top", "#66549A");
arrow(topic2, nodelet, "right", "left", "#C66A18");

textBox("开源边界", { left: 1040, top: 600, width: 154, height: 22 }, 13, "#66549A", true, "center");
textBox("硬件内部实现以编译库提供", { left: 1018, top: 620, width: 198, height: 20 }, 12, "#77708A", false, "center");

s.speakerNotes.textFrame.setText([
  "源码依据：",
  "/Users/jiayiwang/TUM/MA/kuavo_learning_studio/kuavo_deploy/kuavo_env/KuavoBaseRosEnv.py",
  "/Users/jiayiwang/TUM/MA/kuavo_learning_studio/kuavo_model/external_models/lingbot-vla/lingbotvla/data/vla_data/utils.py",
  "/Users/jiayiwang/TUM/MA/kuavo-ros-opensource/src/kuavo_humanoid_sdk/kuavo_humanoid_sdk/kuavo/robot.py",
  "/Users/jiayiwang/TUM/MA/kuavo-ros-opensource/src/kuavo_humanoid_sdk/kuavo_humanoid_sdk/kuavo/core/core.py",
  "/Users/jiayiwang/TUM/MA/kuavo-ros-opensource/src/kuavo_humanoid_sdk/kuavo_humanoid_sdk/kuavo/core/ros/control.py",
  "/Users/jiayiwang/TUM/MA/kuavo-ros-opensource/src/humanoid-control/humanoid_controllers/src/humanoidController.cpp",
  "/Users/jiayiwang/TUM/MA/kuavo-ros-opensource/src/humanoid-control/humanoid_controllers/launch/load_kuavo_real.launch",
].join("\n"));

const preview = await p.export({ slide: s, format: "png", scale: 1.5 });
await fs.writeFile(path.join(TMP_DIR, "preview.png"), new Uint8Array(await preview.arrayBuffer()));
const layout = await s.export({ format: "layout" });
await fs.writeFile(path.join(TMP_DIR, "layout.json"), await layout.text());

const { finalizePresentation } = await import(pathToFileURL(
  path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")
).href);
const candidatePath = path.join(TMP_DIR, "candidate.pptx");
await (await PresentationFile.exportPptx(p)).save(candidatePath);
const result = await finalizePresentation({
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-heading-fit"],
  fontPolicy: { basis: "design", families: [family] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(TMP_DIR, "validation-v2.json"),
});
console.log(JSON.stringify(result, null, 2));
