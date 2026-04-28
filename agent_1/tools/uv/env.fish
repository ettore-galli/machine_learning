if not contains "$HOME/Documents/SVILUPPO/machine_learning/agent_1/tools/uv" $PATH
    # Prepending path in case a system-installed binary needs to be overridden
    set -x PATH "$HOME/Documents/SVILUPPO/machine_learning/agent_1/tools/uv" $PATH
end
