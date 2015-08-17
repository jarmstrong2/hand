require 'nn'

local YHat, parent = torch.class('nn.YHat', 'nn.Module')

function YHat:__init(nGMM)
    self.nGMM = nGMM
end

function YHat:updateOutput(input)
    local split_idx = 1
    local hat_e_t = input[{{},{1}}]
    split_idx = split_idx + 1
    local hat_pi_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    split_idx = split_idx + self.nGMM
    local hat_mu_1_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    split_idx = split_idx + self.nGMM
    local hat_mu_2_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    split_idx = split_idx + self.nGMM
    local hat_sigma_1_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    split_idx = split_idx + self.nGMM
    local hat_sigma_2_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    split_idx = split_idx + self.nGMM
    local hat_rho_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]

    if input:type() == 'torch.CudaTensor' then
        self.useGPU = true
    end

    self.e_t_act = self.e_t_act or nn.Sigmoid()
    self.pi_t_act = self.pi_t_act or nn.SoftMax()
    self.sigma_1_t_act = self.sigma_1_t_act or nn.Exp()
    self.sigma_2_t_act = self.sigma_2_t_act or nn.Exp()
    self.rho_t_act = self.rho_t_act or nn.Tanh()
    if self.useGPU == true then
        self.e_t_act:cuda()
        self.pi_t_act:cuda()
        self.sigma_1_t_act:cuda()
        self.sigma_2_t_act:cuda()
        self.rho_t_act:cuda()
    end

    local e_t = self.e_t_act:forward(-hat_e_t)
    local pi_t = self.pi_t_act:forward(hat_pi_t)
    local mu_1_t = hat_mu_1_t:clone()
    local mu_2_t =  hat_mu_2_t:clone()
    local sigma_1_t = self.sigma_1_t_act:forward(hat_sigma_1_t)
    local sigma_2_t = self.sigma_2_t_act:forward(hat_sigma_2_t)
    local rho_t = self.rho_t_act:forward(hat_rho_t)

    local concat = nn.JoinTable(2)
    if self.useGPU == true then
        concat:cuda()
    end
    self.output = concat:forward({e_t, pi_t, mu_1_t, mu_2_t, sigma_1_t, sigma_2_t, rho_t})

    return self.output
end

function YHat:updateGradInput(input, gradOutput)
    --local split_idx = 1
    --local hat_e_t = input[{{},{1}}]
    --local d_e_t = gradOutput[{{},{1}}]
    --split_idx = split_idx + 1
    --local hat_pi_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    --local d_pi_t = gradOutput[{{},{split_idx,split_idx+self.nGMM-1}}]
    --split_idx = split_idx + self.nGMM
    --local hat_mu_1_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    --local d_mu_1_t = gradOutput[{{},{split_idx,split_idx+self.nGMM-1}}]
    --split_idx = split_idx + self.nGMM
    --local hat_mu_2_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    --local d_mu_2_t = gradOutput[{{},{split_idx,split_idx+self.nGMM-1}}]
    --split_idx = split_idx + self.nGMM
    --local hat_sigma_1_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    --local d_sigma_1_t = gradOutput[{{},{split_idx,split_idx+self.nGMM-1}}]
    --split_idx = split_idx + self.nGMM
    --local hat_sigma_2_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    --local d_sigma_2_t = gradOutput[{{},{split_idx,split_idx+self.nGMM-1}}]
    --split_idx = split_idx + self.nGMM
    --local hat_rho_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    --local d_rho_t = gradOutput[{{},{split_idx,split_idx+self.nGMM-1}}]


    --local grad_e_t = d_e_t:clone() --self.e_t_act:backward(-hat_e_t, d_e_t)
    --local grad_pi_t = d_pi_t:clone() --self.pi_t_act:backward(hat_pi_t, d_pi_t)
    --local grad_mu_1_t = d_mu_1_t:clone()
    --local grad_mu_2_t =  d_mu_2_t:clone()
    --local grad_sigma_1_t = d_sigma_1_t:clone() --self.sigma_1_t_act:backward(hat_sigma_1_t, d_sigma_1_t)
    --local grad_sigma_2_t = d_sigma_2_t:clone() --self.sigma_2_t_act:backward(hat_sigma_2_t, d_sigma_2_t)
    --local grad_rho_t = d_rho_t:clone() --self.rho_t_act:backward(hat_rho_t, d_rho_t)
    --    
    --local grad_input = torch.cat(grad_e_t:float(), grad_pi_t:float())
    --grad_input = torch.cat(grad_input, grad_mu_1_t:float())
    --grad_input = torch.cat(grad_input, grad_mu_2_t:float())
    --grad_input = torch.cat(grad_input, grad_sigma_1_t:float())
    --grad_input = torch.cat(grad_input, grad_sigma_2_t:float())
    --grad_input = torch.cat(grad_input, grad_rho_t:float())
    --
    --self.gradInput = grad_input:cuda()
    self.gradInput = gradOutput
    
    return self.gradInput  
end

