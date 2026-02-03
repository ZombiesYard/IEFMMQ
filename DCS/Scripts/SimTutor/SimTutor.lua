SimTutor = SimTutor or {}

local function _safe_log(message)
    if log and log.write and log.INFO then
        pcall(function()
            log.write("SIMTUTOR", log.INFO, message)
        end)
        return
    end
    if env and env.info then
        pcall(function()
            env.info("SIMTUTOR " .. tostring(message))
        end)
    end
end

local function _safe_error(message)
    if log and log.write and log.ERROR then
        pcall(function()
            log.write("SIMTUTOR", log.ERROR, message)
        end)
        return
    end
    if env and env.info then
        pcall(function()
            env.info("SIMTUTOR ERROR " .. tostring(message))
        end)
    end
end

local function _load_functions()
    local ok, err = pcall(function()
        local st_lfs = require("lfs")
        dofile(st_lfs.writedir() .. "Scripts/SimTutor/SimTutor Function.lua")
    end)
    if ok then
        return true
    end
    _safe_error("Failed to load SimTutor Function.lua from Saved Games: " .. tostring(err))
    local ok2, err2 = pcall(function()
        dofile("SimTutor Function.lua")
    end)
    if ok2 then
        return true
    end
    _safe_error("Failed to load local SimTutor Function.lua: " .. tostring(err2))
    return false
end

local ok, err = pcall(function()
    _safe_log("Initializing ...")
    if _load_functions() and SimTutor.install_export_chain then
        SimTutor.install_export_chain()
    end
    _safe_log("Initialized")
end)

if not ok then
    _safe_error("Initialization failed: " .. tostring(err))
end

