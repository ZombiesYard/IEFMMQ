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

local CMD_HOST, CMD_PORT = "127.0.0.1", 7781
local ACK_HOST, ACK_PORT = "127.0.0.1", 7782
-- AUTO_CLEAR clears any existing highlight before each new highlight (even if target is unchanged).
-- This differs from the Python sender which only clears when switching targets.
local AUTO_CLEAR = true
local HILITE_ID = 9101

local udp_cmd = assert(socket.udp())
assert(udp_cmd:setsockname(CMD_HOST, CMD_PORT))
udp_cmd:settimeout(0)

local udp_ack = assert(socket.udp())
udp_ack:settimeout(0)

logi(("Listening UDP on %s:%d"):format(CMD_HOST, CMD_PORT))

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
