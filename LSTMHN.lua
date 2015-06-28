-- adapted from: wojciechz/learning_to_execute on github

local LSTMHN = {}

-- Creates one timestep of one LSTM
function LSTMHN.lstm()
    local x = nn.Identity()()
    local w = nn.Identity()()
    local below_h = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    function new_input_sum()
        -- transforms input
        local i2h            = nn.Linear(3, 400)(x)
        -- transforms window
        local w2h            = nn.Linear(57, 400)(w)
        -- transforms hidden output from below current hidden layer
        local bh2h            = nn.Linear(400, 400)(below_h)
        -- transforms previous timestep's output
        local h2h            = nn.Linear(400, 400)(prev_h)
        return nn.CAddTable()({i2h, w2h, bh2h, h2h})
    end

    local in_gate          = nn.Sigmoid()(new_input_sum())
    local forget_gate      = nn.Sigmoid()(new_input_sum())
    local out_gate         = nn.Sigmoid()(new_input_sum())
    local in_transform     = nn.Tanh()(new_input_sum())

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return nn.gModule({x, w, below_h, prev_c, prev_h}, {next_c, next_h})
end

return LSTMHN

