-- Saved Games\DCS\Scripts\Hooks\SimTutorHighlight.lua
local TAG = "SIMTUTOR_HILITE"

local function logi(msg) if log and log.write then log.write(TAG, log.INFO, msg) end end
local function loge(msg) if log and log.write then log.write(TAG, log.ERROR, msg) end end

-- LuaSocket
local ok_lfs, lfs = pcall(require, "lfs")
if ok_lfs and lfs and lfs.currentdir then
  local dcsRoot = lfs.currentdir()
  package.path  = package.path .. ";" .. dcsRoot .. "/Scripts/?.lua;" .. dcsRoot .. "/Scripts/?/init.lua"
  package.cpath = package.cpath .. ";" .. dcsRoot .. "/bin/?.dll;" .. dcsRoot .. "/bin-mt/?.dll"
end

local ok_socket, socket = pcall(require, "socket")
if not ok_socket then
  loge("LuaSocket not available: " .. tostring(socket))
  return
end

local ok_json, JSON = pcall(function() return loadfile("Scripts/JSON.lua")() end)
if not ok_json then
  loge("JSON.lua not available: " .. tostring(JSON))
  return
end

local DEFAULT_OVERLAY = {
  command_host = "127.0.0.1",
  command_port = 7781,
  ack_host = "127.0.0.1",
  ack_port = 7782,
  auto_clear = true,
  hilite_id = 9101,
}

local function string_or_default(value, fallback)
  if type(value) == "string" and value ~= "" then
    return value
  end
  return fallback
end

local function number_or_default(value, fallback)
  if type(value) == "number" and value > 0 then
    return math.floor(value)
  end
  return fallback
end

local function load_overlay_config()
  local overlay = {}
  for key, value in pairs(DEFAULT_OVERLAY) do
    overlay[key] = value
  end

  if not ok_lfs or not lfs or not lfs.writedir then
    return overlay
  end

  local cfg_path = lfs.writedir() .. "Scripts\\SimTutor\\SimTutorConfig.lua"
  local ok_cfg, cfg = pcall(function()
    return dofile(cfg_path)
  end)
  if not ok_cfg then
    logi("SimTutorConfig.lua not loaded for overlay settings: " .. tostring(cfg))
    return overlay
  end
  if type(cfg) ~= "table" or type(cfg.overlay) ~= "table" then
    return overlay
  end

  local loaded = cfg.overlay
  overlay.command_host = string_or_default(loaded.command_host, overlay.command_host)
  overlay.command_port = number_or_default(loaded.command_port, overlay.command_port)
  overlay.ack_host = string_or_default(loaded.ack_host, overlay.ack_host)
  overlay.ack_port = number_or_default(loaded.ack_port, overlay.ack_port)
  if type(loaded.auto_clear) == "boolean" then
    overlay.auto_clear = loaded.auto_clear
  end
  if type(loaded.hilite_id) == "number" and loaded.hilite_id >= 0 then
    overlay.hilite_id = math.floor(loaded.hilite_id)
  end
  return overlay
end

local OVERLAY = load_overlay_config()
local CMD_HOST, CMD_PORT = OVERLAY.command_host, OVERLAY.command_port
local ACK_HOST, ACK_PORT = OVERLAY.ack_host, OVERLAY.ack_port
-- AUTO_CLEAR clears any existing highlight before each new highlight (even if target is unchanged).
-- This differs from the Python sender which only clears when switching targets.
local AUTO_CLEAR = OVERLAY.auto_clear
local HILITE_ID = OVERLAY.hilite_id

local udp_cmd = assert(socket.udp())
assert(udp_cmd:setsockname(CMD_HOST, CMD_PORT))
udp_cmd:settimeout(0)

local udp_ack = assert(socket.udp())
udp_ack:settimeout(0)

logi(
  ("Listening UDP on %s:%d; ACK -> %s:%d; auto_clear=%s; hilite_id=%d"):format(
    CMD_HOST,
    CMD_PORT,
    ACK_HOST,
    ACK_PORT,
    tostring(AUTO_CLEAR),
    HILITE_ID
  )
)

local function missionEval(chunk)
  if not net or not net.dostring_in then
    return nil, "net.dostring_in not available"
  end
  local ok, res = pcall(net.dostring_in, "mission", chunk)
  if not ok then
    return nil, tostring(res)
  end
  return res, nil
end

local function as_lua_string(value)
  return string.format("%q", tostring(value))
end

local function send_ack(cmd_id, status, reason)
  local payload = {
    schema_version = "v2",
    cmd_id = cmd_id,
    status = status,
  }
  if reason ~= nil then
    payload.reason = reason
  end
  local ok, json_str = pcall(function() return JSON:encode(payload) end)
  if not ok then
    loge("Failed to encode ack: " .. tostring(json_str))
    return
  end
  pcall(function()
    udp_ack:sendto(json_str, ACK_HOST, ACK_PORT)
  end)
end

local function do_highlight(pnt)
  if AUTO_CLEAR then
    missionEval(('a_cockpit_remove_highlight(%d)'):format(HILITE_ID))
  end
  local code = ('a_cockpit_highlight(%d, %s, 0, "")'):format(HILITE_ID, as_lua_string(pnt))
  local _, err = missionEval(code)
  if err then
    return false, err
  end
  return true, nil
end

local function do_clear()
  local _, err = missionEval(('a_cockpit_remove_highlight(%d)'):format(HILITE_ID))
  if err then
    return false, err
  end
  return true, nil
end

local function handle_command(cmd)
  local cmd_id = cmd.cmd_id
  local action = cmd.action
  local target = cmd.target
  if not cmd_id or type(cmd_id) ~= "string" then
    loge("Invalid command: missing/invalid cmd_id")
    return
  end
  if action == "clear" then
    local ok, err = do_clear()
    send_ack(cmd_id, ok and "ok" or "failed", err)
    return
  end
  if action == "highlight" and target then
    local ok, err = do_highlight(target)
    send_ack(cmd_id, ok and "ok" or "failed", err)
    return
  end
  send_ack(cmd_id, "failed", "invalid command")
end

local function parse_json(data)
  local ok, obj = pcall(function() return JSON:decode(data) end)
  if not ok or type(obj) ~= "table" then
    return nil, "invalid json"
  end
  return obj, nil
end

local callbacks = {}

function callbacks.onSimulationFrame()
  while true do
    local data = udp_cmd:receive()
    if not data then break end
    local cmd, err = parse_json(data)
    if cmd then
      handle_command(cmd)
    else
      loge("Invalid command: " .. tostring(err))
    end
  end
end

if DCS and DCS.setUserCallbacks then
  DCS.setUserCallbacks(callbacks)
  logi("User callbacks registered (onSimulationFrame).")
else
  loge("DCS.setUserCallbacks not available - script may be running in unexpected env.")
end
