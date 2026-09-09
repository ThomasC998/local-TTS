--------------------------------------------------------------------------
-- Breeze TTS: speak whatever is on the clipboard.
--
-- This file is deliberately thin. Hammerspoon owns the two things a browser
-- cannot do -- a system-wide hotkey and a clipboard read while another app is
-- frontmost -- and posts the text to the local server. Voice selection, the
-- language-model pass, tone rotation, paragraph seeking and playback all happen
-- there, so none of it has to be configured twice.
--
--   ctrl-alt-S   speak the clipboard -- press again to stop
--   ctrl-alt-A   the same, through the language model first
--   ctrl-alt-→   skip to the next paragraph
--   ctrl-alt-←   go back a paragraph
--   ctrl-alt-X   stop
--
-- Those are the defaults. The live set comes from the server, edited on the
-- System speech tab of the web UI, so the Mac and the Windows machine share
-- one configuration. After changing them there, reload this config --
-- `open -g "hammerspoon://breeze-reload"` -- to pick them up.
--
-- S and A are toggles. One press starts the read, the next silences it, and the
-- press after that starts a new read from whatever is on the clipboard by then
-- -- so copying something else and hitting the same key twice is all it takes.
--
-- The skip keys repeat while held, and the server debounces them: nothing is
-- generated until the presses stop, so holding one scrolls through the document
-- instead of synthesizing every paragraph on the way past. A skip press also
-- resumes a read that was paused because the output device went away, which is
-- what happens when Bluetooth headphones run out of battery.
--
-- The arrow keys are used because they are in the same place on every layout.
-- Rectangle claims ctrl-alt with the arrows in its *default* shortcut set, so if
-- window snapping stops working, switch Rectangle to its alternate shortcuts
-- (Preferences -> "Use alternate default shortcuts") or change the seek keys on
-- the System speech tab.
--------------------------------------------------------------------------

local M = {}

M.server = "http://127.0.0.1:7860"
-- nil means "use the voice chosen on the System speech tab of the web UI".
M.voiceId = nil
-- Start Hammerspoon at login, so the hotkeys survive a reboot. Off by default:
-- adding a login item is not something a hotkey installer should do unasked.
-- `python install_hotkeys.py --launch-agent` turns it on, along with installing
-- the server's own login agent.
M.autoLaunch = false

local function alert(message)
  hs.alert.closeAll(0)
  hs.alert.show(message, 1.2)
end

local function post(path, body, onOk)
  hs.http.asyncPost(
    M.server .. path,
    hs.json.encode(body or {}),
    { ["Content-Type"] = "application/json" },
    function(status, response)
      if status == 200 then
        local ok, parsed = pcall(hs.json.decode, response or "")
        if onOk then onOk(ok and parsed or nil) end
      elseif status <= 0 then
        alert("Breeze server is not running")
      else
        local detail = response
        local ok, parsed = pcall(hs.json.decode, response or "")
        if ok and parsed and parsed.detail then detail = parsed.detail end
        alert("Breeze: " .. tostring(detail))
      end
    end
  )
end

-- One key, both directions. The server decides which it is, because only the
-- server knows whether anything is still playing -- asking it first would race
-- against a read that ended in between.
function M.toggle(useLlm)
  local text = hs.pasteboard.getContents()
  if not text or text:gsub("%s", "") == "" then
    -- Still worth sending: an empty clipboard cannot start a read, but the
    -- press may well have been meant to stop one.
    post("/v1/speak/toggle", {}, function(result)
      if result and result.action == "stopped" then
        alert("Stopped")
      else
        alert("Clipboard is empty")
      end
    end)
    return
  end
  post(
    "/v1/speak/toggle",
    { text = text, llm = useLlm or false, voice_id = M.voiceId },
    function(result)
      if result and result.action == "stopped" then
        alert("Stopped")
      else
        alert(useLlm and "Reading (model)…" or "Reading…")
      end
    end
  )
end

function M.skip(delta)
  post("/v1/speak/skip", { delta = delta }, function(result)
    if not result or result.action ~= "skipped" then
      alert("Nothing is being read")
      return
    end
    local target = result.paragraph_target or result.paragraph or 0
    local total = result.paragraphs or 0
    local position = string.format("¶ %d", target + 1)
    if result.paragraphs_final and total > 0 then
      position = string.format("¶ %d of %d", target + 1, total)
    end
    alert((delta < 0 and "◀ " or "▶ ") .. position)
  end)
end

function M.stop()
  post("/v1/speak/stop", {}, function() alert("Stopped") end)
end

--------------------------------------------------------------------------
-- Binding
--
-- The shortcuts come from the server, which is also where the web UI edits
-- them, so the Mac and the Windows machine cannot drift apart. Two rounds:
-- the defaults are bound immediately so the keys work even with the server
-- down, then the configured set replaces them when the server answers.
--
-- Rebinding means deleting the old hotkeys first. Hammerspoon happily binds
-- the same combination twice, and then every press fires both handlers.
--------------------------------------------------------------------------

-- Matches hotkeys/bindings.py. "cmd" is written that way in the config and is
-- the Windows key over there; here it is Command.
M.defaults = {
  toggle = "ctrl+alt+s",
  toggle_llm = "ctrl+alt+a",
  next = "ctrl+alt+right",
  previous = "ctrl+alt+left",
  stop = "ctrl+alt+x",
}

local bound = {}

local function parse(binding)
  local mods, key = {}, nil
  for part in tostring(binding):gmatch("[^+]+") do
    part = part:lower():gsub("^%s*(.-)%s*$", "%1")
    if part == "ctrl" or part == "control" then table.insert(mods, "ctrl")
    elseif part == "alt" or part == "option" then table.insert(mods, "alt")
    elseif part == "shift" then table.insert(mods, "shift")
    elseif part == "cmd" or part == "command" or part == "win" then
      table.insert(mods, "cmd")
    else key = part end
  end
  if not key or #mods == 0 then return nil end
  return mods, key
end

local function bindOne(binding, handler, repeats)
  local mods, key = parse(binding)
  if not mods then
    alert("Breeze: cannot bind " .. tostring(binding))
    return
  end
  -- The fifth argument is the auto-repeat handler, so holding a skip key
  -- scrolls the document instead of needing one press per paragraph.
  local hotkey = hs.hotkey.bind(mods, key, handler, nil,
                                repeats and handler or nil)
  table.insert(bound, hotkey)
end

function M.bind(keys)
  keys = keys or M.defaults
  for _, hotkey in ipairs(bound) do hotkey:delete() end
  bound = {}
  bindOne(keys.toggle or M.defaults.toggle, function() M.toggle(false) end)
  bindOne(keys.toggle_llm or M.defaults.toggle_llm, function() M.toggle(true) end)
  bindOne(keys.stop or M.defaults.stop, M.stop)
  bindOne(keys.next or M.defaults.next, function() M.skip(1) end, true)
  bindOne(keys.previous or M.defaults.previous, function() M.skip(-1) end, true)
end

-- Ask the server what the shortcuts are, and rebind if it has an opinion.
-- Silent on failure: the defaults are already bound, and a modal alert every
-- time Hammerspoon reloads while the server is down would be worse than the
-- keys quietly being the standard ones.
function M.syncBindings()
  hs.http.asyncGet(M.server .. "/v1/hotkeys", nil, function(status, response)
    if status ~= 200 then return end
    local ok, parsed = pcall(hs.json.decode, response or "")
    if ok and parsed and parsed.bindings then M.bind(parsed.bindings) end
  end)
end

-- Lets `open -g "hammerspoon://breeze-reload"` pick up config changes without
-- reaching for the menu bar.
hs.urlevent.bind("breeze-reload", function() hs.reload() end)

if M.autoLaunch then hs.autoLaunch(true) end

M.bind()
M.syncBindings()

return M
