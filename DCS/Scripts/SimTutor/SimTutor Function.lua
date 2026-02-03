SimTutor = SimTutor or {}

SimTutor.VERSION = "0.2.0"
SimTutor._export_installed = SimTutor._export_installed or false

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
        SimTutor.log_info("LuaExportStart")
    end

    function LuaExportBeforeNextFrame()
        SimTutor.safe_call(upstream_before, "upstream LuaExportBeforeNextFrame")
    end

    function LuaExportAfterNextFrame()
        SimTutor.safe_call(upstream_after, "upstream LuaExportAfterNextFrame")
    end

    function LuaExportStop()
        SimTutor.safe_call(upstream_stop, "upstream LuaExportStop")
        SimTutor.log_info("LuaExportStop")
    end
end

