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
  hilite_ids = nil,
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

local function clone_array(raw)
  local out = {}
  if type(raw) ~= "table" then
    return out
  end
  for idx, value in ipairs(raw) do
    out[idx] = value
  end
  return out
end

local function normalize_hilite_ids(raw, fallback_id)
  local ids = {}
  local seen = {}
  if type(raw) == "table" then
    for _, value in ipairs(raw) do
      if type(value) == "number" and value >= 0 then
        local normalized = math.floor(value)
        if not seen[normalized] then
          seen[normalized] = true
          ids[#ids + 1] = normalized
        end
      end
    end
  end
  if #ids == 0 and type(fallback_id) == "number" and fallback_id >= 0 then
    ids[1] = math.floor(fallback_id)
  end
  return ids
end

local function load_overlay_config()
  local overlay = {}
  for key, value in pairs(DEFAULT_OVERLAY) do
    if key == "hilite_ids" then
      overlay[key] = clone_array(value)
    else
      overlay[key] = value
    end
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
  if type(loaded.hilite_ids) == "table" then
    overlay.hilite_ids = clone_array(loaded.hilite_ids)
  end
  overlay.hilite_ids = normalize_hilite_ids(overlay.hilite_ids, overlay.hilite_id)
  return overlay
end

local OVERLAY = load_overlay_config()
local CMD_HOST, CMD_PORT = OVERLAY.command_host, OVERLAY.command_port
local ACK_HOST, ACK_PORT = OVERLAY.ack_host, OVERLAY.ack_port
-- AUTO_CLEAR only matters when no free highlight slot exists for a new target.
local AUTO_CLEAR = OVERLAY.auto_clear
local HILITE_ID = OVERLAY.hilite_id
local HILITE_IDS = normalize_hilite_ids(OVERLAY.hilite_ids, HILITE_ID)
local ACTIVE_BY_TARGET = {}
local ACTIVE_ORDER = {}

local udp_cmd = assert(socket.udp())
assert(udp_cmd:setsockname(CMD_HOST, CMD_PORT))
udp_cmd:settimeout(0)

local udp_ack = assert(socket.udp())
udp_ack:settimeout(0)

logi(
  ("Listening UDP on %s:%d; ACK -> %s:%d; auto_clear=%s; hilite_ids=%s"):format(
    CMD_HOST,
    CMD_PORT,
    ACK_HOST,
    ACK_PORT,
    tostring(AUTO_CLEAR),
    table.concat(HILITE_IDS, ",")
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

local function remove_active_order(target)
  for idx = #ACTIVE_ORDER, 1, -1 do
    if ACTIVE_ORDER[idx] == target then
      table.remove(ACTIVE_ORDER, idx)
    end
  end
end

local function clear_hilite_id(hilite_id)
  local _, err = missionEval(('a_cockpit_remove_highlight(%d)'):format(hilite_id))
  if err then
    return false, err
  end
  return true, nil
end

local function clear_target(target)
  local active = ACTIVE_BY_TARGET[target]
  if type(active) ~= "table" then
    return true, nil
  end
  local ok, err = clear_hilite_id(active.hilite_id)
  if not ok then
    return false, err
  end
  ACTIVE_BY_TARGET[target] = nil
  remove_active_order(target)
  return true, nil
end

local function clear_all_targets()
  local first_err = nil
  for _, target in ipairs(clone_array(ACTIVE_ORDER)) do
    local ok, err = clear_target(target)
    if not ok and first_err == nil then
      first_err = err
    end
  end
  if first_err ~= nil then
    return false, first_err
  end
  return true, nil
end

local function find_free_hilite_id()
  local used = {}
  for _, active in pairs(ACTIVE_BY_TARGET) do
    if type(active) == "table" and type(active.hilite_id) == "number" then
      used[active.hilite_id] = true
    end
  end
  for _, hilite_id in ipairs(HILITE_IDS) do
    if not used[hilite_id] then
      return hilite_id
    end
  end
  return nil
end

local function assign_hilite_id_for_target(target)
  local current = ACTIVE_BY_TARGET[target]
  if type(current) == "table" and type(current.hilite_id) == "number" then
    remove_active_order(target)
    ACTIVE_ORDER[#ACTIVE_ORDER + 1] = target
    return current.hilite_id, nil
  end

  local free_id = find_free_hilite_id()
  if free_id ~= nil then
    ACTIVE_ORDER[#ACTIVE_ORDER + 1] = target
    return free_id, nil
  end

  if AUTO_CLEAR then
    local ok, err = clear_all_targets()
    if not ok then
      return nil, err
    end
    ACTIVE_ORDER[#ACTIVE_ORDER + 1] = target
    return HILITE_IDS[1], nil
  end

  return nil, "no free highlight slot"
end

local function do_highlight(pnt)
  local hilite_id, assign_err = assign_hilite_id_for_target(pnt)
  if not hilite_id then
    return false, assign_err
  end
  local code = ('a_cockpit_highlight(%d, %s, 0, "")'):format(hilite_id, as_lua_string(pnt))
  local _, err = missionEval(code)
  if err then
    return false, err
  end
  ACTIVE_BY_TARGET[pnt] = { hilite_id = hilite_id }
  return true, nil
end

local function do_clear(target)
  if type(target) == "string" and target ~= "" then
    return clear_target(target)
  end
  return clear_all_targets()
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
    local ok, err = do_clear(target)
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
