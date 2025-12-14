-- Saved Games\DCS\Scripts\Hooks\VRHILITE.lua
local TAG = "VRHILITE"

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

local HOST, PORT = "127.0.0.1", 7778
local udp = assert(socket.udp())
assert(udp:setsockname(HOST, PORT))
udp:settimeout(0)
logi(("Listening UDP on %s:%d"):format(HOST, PORT))

local HILITE_ID = "9001"

local function missionEval(chunk)
  if not net or not net.dostring_in then
    loge("net.dostring_in not available (likely wrong env / MP restriction).")
    return nil
  end
  local ok, res = pcall(net.dostring_in, "mission", chunk)
  if not ok then
    loge("net.dostring_in failed: " .. tostring(res))
    return nil
  end
  return res
end

local function highlight(pnt)
  -- 先清一次，避免重复叠加/状态卡住
  missionEval(('a_cockpit_remove_highlight(%d)'):format(HILITE_ID))

  local code = ('a_cockpit_highlight(%d, "%s", 0, "")'):format(HILITE_ID, pnt)
  missionEval(code)
  logi("Highlighted " .. tostring(pnt) .. " (id=" .. tostring(HILITE_ID) .. ")")
end

local function clear()
  missionEval(('a_cockpit_remove_highlight(%d)'):format(HILITE_ID))
  logi("Cleared highlight (id=" .. tostring(HILITE_ID) .. ")")
end

local function handleLine(line)
  line = tostring(line):gsub("%s+$","")
  if line == "" then return end

  if line == "CLEAR" then
    clear()
    return
  end

  -- 支持两种：直接发 "pnt_331" 或 "HILITE pnt_331"
  local pnt = line:match("^HILITE%s+(.+)$") or line
  highlight(pnt)
end

-- Hook 回调：每帧轮询 UDP（非阻塞）
local callbacks = {}

function callbacks.onSimulationFrame()
  while true do
    local data = udp:receive()
    if not data then break end
    handleLine(data)
  end
end

if DCS and DCS.setUserCallbacks then
  DCS.setUserCallbacks(callbacks)
  logi("User callbacks registered (onSimulationFrame).")
else
  loge("DCS.setUserCallbacks not available - script may be running in unexpected env.")
end
