SimTutor = SimTutor or {}

SimTutor.VERSION = "0.2.0"
SimTutor._export_installed = SimTutor._export_installed or false
SimTutor.telemetry = SimTutor.telemetry or {
    enabled = true,
    host = "127.0.0.1",
    port = 7780,
    hz = 20,
    include_raw = false,
    debug_log = false,
}
SimTutor.caps = SimTutor.caps or {
    telemetry = true,
    overlay = false,
    overlay_ack = false,
    clickable_actions = false,
    vlm_frame = false,
}
SimTutor.handshake = SimTutor.handshake or {
    enabled = true,
    host = "127.0.0.1",
    port = 7793,
}
SimTutor._telemetry = SimTutor._telemetry or {
    socket = nil,
    socket_lib = nil,
    json = nil,
    last_send = 0,
    seq = 0,
    last_error_log = 0,
}
SimTutor._handshake = SimTutor._handshake or {
    socket = nil,
    socket_lib = nil,
    json = nil,
    last_error_log = 0,
}

local function _safe_log(level, message)
    if log and log.write and log.INFO and log.ERROR then
        pcall(function()
            log.write("SIMTUTOR", level, message)
        end)
        return
    end
    if env and env.info then
        pcall(function()
            env.info("SIMTUTOR " .. tostring(message))
        end)
    end
end

function SimTutor.log_info(message)
    _safe_log(log and log.INFO or 1, message)
end

function SimTutor.log_error(message)
    _safe_log(log and log.ERROR or 3, message)
end

function SimTutor.safe_call(fn, label)
    if not fn then
        return
    end
    local ok, err = pcall(fn)
    if not ok then
        SimTutor.log_error("Error in " .. tostring(label) .. ": " .. tostring(err))
    end
end

function SimTutor.install_export_chain()
    if SimTutor._export_installed then
        return
    end
    SimTutor._export_installed = true

    local upstream_start = LuaExportStart
    local upstream_before = LuaExportBeforeNextFrame
    local upstream_after = LuaExportAfterNextFrame
    local upstream_stop = LuaExportStop

    function LuaExportStart()
        SimTutor.safe_call(upstream_start, "upstream LuaExportStart")
        SimTutor.safe_call(SimTutor._init_telemetry, "SimTutor._init_telemetry")
        SimTutor.safe_call(SimTutor._init_handshake, "SimTutor._init_handshake")
        SimTutor.log_info("LuaExportStart")
    end

    function LuaExportBeforeNextFrame()
        SimTutor.safe_call(upstream_before, "upstream LuaExportBeforeNextFrame")
    end

    function LuaExportAfterNextFrame()
        SimTutor.safe_call(upstream_after, "upstream LuaExportAfterNextFrame")
        SimTutor.safe_call(SimTutor._telemetry_tick, "SimTutor._telemetry_tick")
        SimTutor.safe_call(SimTutor._handshake_tick, "SimTutor._handshake_tick")
    end

    function LuaExportStop()
        SimTutor.safe_call(upstream_stop, "upstream LuaExportStop")
        SimTutor.safe_call(SimTutor._shutdown_telemetry, "SimTutor._shutdown_telemetry")
        SimTutor.safe_call(SimTutor._shutdown_handshake, "SimTutor._shutdown_handshake")
        SimTutor.log_info("LuaExportStop")
    end
end

function SimTutor._init_telemetry()
    if not SimTutor.telemetry.enabled then
        return
    end
    SimTutor._extend_package_paths()
    SimTutor._load_config()
    local ok, socket_lib = pcall(require, "socket")
    if not ok then
        SimTutor.log_error("Telemetry disabled: socket not available")
        SimTutor.telemetry.enabled = false
        return
    end
    local ok_json, json_mod = SimTutor._load_json()
    if not ok_json then
        SimTutor.log_error("Telemetry disabled: JSON.lua not available")
        SimTutor.telemetry.enabled = false
        return
    end
    local udp = socket_lib.udp()
    udp:settimeout(0)
    SimTutor._telemetry.socket = udp
    SimTutor._telemetry.socket_lib = socket_lib
    SimTutor._telemetry.json = json_mod
    SimTutor._telemetry.last_send = 0
    SimTutor._telemetry.seq = 0
    SimTutor._telemetry.last_error_log = 0
    SimTutor.log_info(
        string.format(
            "Telemetry initialized host=%s port=%s hz=%s",
            tostring(SimTutor.telemetry.host),
            tostring(SimTutor.telemetry.port),
            tostring(SimTutor.telemetry.hz)
        )
    )
end

function SimTutor._shutdown_telemetry()
    local udp = SimTutor._telemetry.socket
    if udp then
        pcall(function()
            udp:close()
        end)
    end
    SimTutor._telemetry.socket = nil
    SimTutor._telemetry.socket_lib = nil
end

function SimTutor._init_handshake()
    if not SimTutor.handshake.enabled then
        return
    end
    SimTutor._extend_package_paths()
    SimTutor._load_config()
    local ok, socket_lib = pcall(require, "socket")
    if not ok then
        SimTutor.log_error("Handshake disabled: socket not available")
        SimTutor.handshake.enabled = false
        return
    end
    local ok_json, json_mod = SimTutor._load_json()
    if not ok_json then
        SimTutor.log_error("Handshake disabled: JSON.lua not available")
        SimTutor.handshake.enabled = false
        return
    end
    local udp = socket_lib.udp()
    udp:setsockname(SimTutor.handshake.host, SimTutor.handshake.port)
    udp:settimeout(0)
    SimTutor._handshake.socket = udp
    SimTutor._handshake.socket_lib = socket_lib
    SimTutor._handshake.json = json_mod
    SimTutor._handshake.last_error_log = 0
    SimTutor.log_info(
        string.format(
            "Handshake listening host=%s port=%s",
            tostring(SimTutor.handshake.host),
            tostring(SimTutor.handshake.port)
        )
    )
end

function SimTutor._shutdown_handshake()
    local udp = SimTutor._handshake.socket
    if udp then
        pcall(function()
            udp:close()
        end)
    end
    SimTutor._handshake.socket = nil
    SimTutor._handshake.socket_lib = nil
end

function SimTutor._build_caps()
    local caps = {
        schema_version = "v2",
        telemetry = (SimTutor.telemetry.enabled and SimTutor.caps.telemetry) and true or false,
        overlay = SimTutor.caps.overlay and true or false,
        overlay_ack = SimTutor.caps.overlay_ack and true or false,
        clickable_actions = SimTutor.caps.clickable_actions and true or false,
        vlm_frame = SimTutor.caps.vlm_frame and true or false,
    }
    if not caps.overlay then
        caps.overlay_ack = false
    end
    return caps
end

function SimTutor._handle_hello(payload, addr, port)
    if not payload or type(payload) ~= "table" then
        return
    end
    if payload.client ~= "simtutor" then
        return
    end
    local caps = SimTutor._build_caps()
    local ok, json_str = pcall(function() return SimTutor._handshake.json:encode(caps) end)
    if not ok then
        SimTutor._handshake_log_error("Failed to encode caps: " .. tostring(json_str))
        return
    end
    pcall(function()
        SimTutor._handshake.socket:sendto(json_str, addr, port)
    end)
end

function SimTutor._handshake_tick()
    if not SimTutor.handshake.enabled then
        return
    end
    local udp = SimTutor._handshake.socket
    local json = SimTutor._handshake.json
    if not udp or not json then
        return
    end
    while true do
        local data, addr, port = udp:receivefrom()
        if not data then
            break
        end
        local ok, payload = pcall(function() return json:decode(data) end)
        if ok and payload then
            SimTutor._handle_hello(payload, addr, port)
        else
            SimTutor._handshake_log_error("Invalid hello payload")
        end
    end
end

local function _set_if_number(target, key, value)
    if type(value) == "number" then
        target[key] = value
    end
end

function SimTutor._build_cockpit_snapshot()
    local cockpit = {}
    local self_data = nil
    pcall(function()
        self_data = LoGetSelfData()
    end)
    if self_data then
        if type(self_data.Name) == "string" then
            cockpit["aircraft_name"] = self_data.Name
        end
        _set_if_number(cockpit, "speed", self_data.Speed)
        _set_if_number(cockpit, "heading", self_data.Heading)
        _set_if_number(cockpit, "pitch", self_data.Pitch)
        _set_if_number(cockpit, "bank", self_data.Bank)
        _set_if_number(cockpit, "altitude", self_data.Alt)
        _set_if_number(cockpit, "radar_altitude", self_data.RadarAltitude)
    end
    return cockpit, self_data
end

function SimTutor._telemetry_tick()
    if not SimTutor.telemetry.enabled then
        return
    end
    local udp = SimTutor._telemetry.socket
    local json = SimTutor._telemetry.json
    local socket_lib = SimTutor._telemetry.socket_lib
    if not udp or not json then
        return
    end
    if not socket_lib then
        return
    end
    local now = socket_lib.gettime()
    local interval = 1 / math.max(1, SimTutor.telemetry.hz or 20)
    if (now - SimTutor._telemetry.last_send) < interval then
        return
    end
    SimTutor._telemetry.last_send = now
    SimTutor._telemetry.seq = SimTutor._telemetry.seq + 1

    local cockpit, self_data = SimTutor._build_cockpit_snapshot()
    local sim_time = nil
    pcall(function()
        sim_time = LoGetModelTime()
    end)
    local aircraft = (self_data and self_data.Name) or "Unknown"

    local frame = {
        schema_version = "v2",
        seq = SimTutor._telemetry.seq,
        sim_time = sim_time or 0,
        aircraft = aircraft,
        cockpit = cockpit,
    }

    if SimTutor.telemetry.include_raw then
        frame["raw"] = {
            self_data = self_data,
        }
    end

    local payload = json:encode(frame)
    local ok, err = pcall(function()
        udp:sendto(payload, SimTutor.telemetry.host, SimTutor.telemetry.port)
    end)
    if not ok then
        SimTutor._telemetry_log_error("Telemetry send failed: " .. tostring(err))
        return
    end
    if SimTutor.telemetry.debug_log then
        SimTutor.log_info("Telemetry sent seq=" .. tostring(frame.seq))
    end
end

function SimTutor._extend_package_paths()
    local ok, st_lfs = pcall(require, "lfs")
    if not ok or not st_lfs then
        return
    end
    local base = st_lfs.currentdir()
    local sep = package.config:sub(1, 1)
    if base:sub(-1) ~= sep then
        base = base .. sep
    end
    local lua_socket_path = base .. "LuaSocket" .. sep .. "?.lua"
    local lua_socket_cpath = base .. "LuaSocket" .. sep .. "?.dll"
    if not string.find(package.path, lua_socket_path, 1, true) then
        package.path = package.path .. ";" .. lua_socket_path
    end
    if not string.find(package.cpath, lua_socket_cpath, 1, true) then
        package.cpath = package.cpath .. ";" .. lua_socket_cpath
    end
    local sg_scripts = st_lfs.writedir() .. "Scripts" .. sep .. "?.lua"
    if not string.find(package.path, sg_scripts, 1, true) then
        package.path = package.path .. ";" .. sg_scripts
    end
end

function SimTutor._load_json()
    local ok_json, json_mod = pcall(function()
        return loadfile("Scripts\\JSON.lua")()
    end)
    if ok_json then
        return ok_json, json_mod
    end
    local ok, st_lfs = pcall(require, "lfs")
    if not ok or not st_lfs then
        return ok_json, json_mod
    end
    local base = st_lfs.currentdir()
    local sep = package.config:sub(1, 1)
    if base:sub(-1) ~= sep then
        base = base .. sep
    end
    local path = base .. "Scripts" .. sep .. "JSON.lua"
    return pcall(function()
        return loadfile(path)()
    end)
end

function SimTutor._load_config()
    local ok, st_lfs = pcall(require, "lfs")
    if not ok or not st_lfs then
        return
    end
    local cfg_path = st_lfs.writedir() .. "Scripts\\SimTutor\\SimTutorConfig.lua"
    local ok_cfg, cfg = pcall(function()
        return dofile(cfg_path)
    end)
    if not ok_cfg or type(cfg) ~= "table" then
        return
    end
    if type(cfg.telemetry) == "table" then
        for k, v in pairs(cfg.telemetry) do
            SimTutor.telemetry[k] = v
        end
    else
        for k, v in pairs(cfg) do
            SimTutor.telemetry[k] = v
        end
    end
    if type(cfg.caps) == "table" then
        for k, v in pairs(cfg.caps) do
            SimTutor.caps[k] = v
        end
    end
    if type(cfg.handshake) == "table" then
        for k, v in pairs(cfg.handshake) do
            SimTutor.handshake[k] = v
        end
    end
end

function SimTutor._telemetry_log_error(message)
    local socket_lib = SimTutor._telemetry.socket_lib
    local now = socket_lib and socket_lib.gettime() or 0
    if (now - SimTutor._telemetry.last_error_log) < 5 then
        return
    end
    SimTutor._telemetry.last_error_log = now
    SimTutor.log_error(message)
end

function SimTutor._handshake_log_error(message)
    local socket_lib = SimTutor._handshake.socket_lib
    local now = socket_lib and socket_lib.gettime() or 0
    if (now - SimTutor._handshake.last_error_log) < 5 then
        return
    end
    SimTutor._handshake.last_error_log = now
    SimTutor.log_error(message)
end
