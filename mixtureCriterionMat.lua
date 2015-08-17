require 'nn'
local MixtureCriterion, parent = torch.class('nn.MixtureCriterion', 'nn.Criterion')

function MixtureCriterion:__init(nGMM)
    self.nGMM = nGMM
end

function MixtureCriterion:setmask(mask)
   self.mask = mask 
end

function MixtureCriterion:setSizeAverage()
   self.sizeAverage = true 
end

function MixtureCriterion:updateOutput(input, target)

    local x1 = target[{{},{1}}]
    local x2 = target[{{},{2}}]
    local x3 = target[{{},{3}}]
    
    if input:type() == 'torch.CudaTensor' then
        self.useGPU = true
    end


    local split_idx = 1
    local e_t = input[{{},{1}}]
    split_idx = split_idx + 1
    local pi_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    split_idx = split_idx + self.nGMM
    local mu_1_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    split_idx = split_idx + self.nGMM
    local mu_2_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    split_idx = split_idx + self.nGMM
    local sigma_1_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    split_idx = split_idx + self.nGMM
    local sigma_2_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    split_idx = split_idx + self.nGMM
    local rho_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]

   
    local sampleSize = (#input)[1]
    
    self.inv_sigma1 = sigma_1_t:clone():fill(1.):cdiv(sigma_1_t + 1e-8)
    self.inv_sigma2 = sigma_2_t:clone():fill(1.):cdiv(sigma_2_t + 1e-8)
    self.mixdist1 = torch.cmul(self.inv_sigma1, self.inv_sigma2)
    self.inv_1mrho2 = rho_t:clone():fill(1.):cdiv((-rho_t:clone():cmul(rho_t)):add(1))
    self.mixdist1:cmul(self.inv_1mrho2:clone():sqrt())
    self.mixdist1:mul(1/(2*math.pi))
   
 
    self.mu_1_x_1 = (mu_1_t:clone()):mul(-1)
    self.mu_2_x_2 = (mu_2_t:clone()):mul(-1)
    local x1_val = x1:expand(sampleSize, self.nGMM)
    local x2_val = x2:expand(sampleSize, self.nGMM) 
    self.mu_1_x_1:add(x1_val)
    self.mu_2_x_2:add(x2_val) 
    
    local mixdist2_z_1 = torch.cmul(torch.pow(self.inv_sigma1, 2), torch.pow(self.mu_1_x_1,2))  
    local mixdist2_z_2 = torch.cmul(torch.pow(self.inv_sigma2, 2), torch.pow(self.mu_2_x_2,2)) 
    local mixdist2_z_3 = torch.cmul(self.inv_sigma1, self.inv_sigma2)
    
    mixdist2_z_3:cmul(self.mu_1_x_1)
    mixdist2_z_3:cmul(self.mu_2_x_2)
    
    mixdist2_z_3:cmul(torch.mul(rho_t, 2))
    local z = mixdist2_z_1 + mixdist2_z_2 - mixdist2_z_3
    self.z = z

    local mixdist2 = z:clone()
    mixdist2:mul(-1)
    mixdist2:cmul(self.inv_1mrho2):div(2.)
    --print('log mixdist2', mixdist2:sum())
    mixdist2:exp()
    self.mixdist = torch.cmul(self.mixdist1, mixdist2)
    self.mixdist:cmul(pi_t)
    
    -- sum of mixture components 
    local mixdist_sum = torch.sum(self.mixdist, 2)
    self.mixdist_sum = mixdist_sum
    local log_mixdist_sum = torch.log(mixdist_sum)
    
    -- compute end of stroke mixing
    local log_e_t = e_t:clone()
    log_e_t:log():cmul(x3):add(e_t:clone():mul(-1):add(1):log():cmul(x3:clone():mul(-1):add(1)))
    --local log_e_t = e_t:clone()
    
    --local eq1 = torch.eq(x3, torch.ones(sampleSize, 1):cuda())
    --eq1 = eq1:cuda()
    --eq1:cmul(torch.log(e_t))
    --local neq1 = torch.ne(x3, torch.ones(sampleSize, 1):cuda())
    --neq1 = neq1:cuda()
    --neq1:cmul(torch.log(-e_t + 1))
    --local log_e_t = eq1 + neq1
    
    local result = log_mixdist_sum + log_e_t
    result:mul(-1)
    result:cmul(self.mask)
    result = result:sum()
    if self.sizeAverage then
        result = result/target:size(1)
    end
    
    return result
end

function MixtureCriterion:updateGradInput(input, target)
    local x1 = target[{{},{1}}]
    local x2 = target[{{},{2}}]
    local x3 = target[{{},{3}}]

    local split_idx = 1
    local e_t = input[{{},{1}}]
    split_idx = split_idx + 1
    local pi_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    split_idx = split_idx + self.nGMM
    local mu_1_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    split_idx = split_idx + self.nGMM
    local mu_2_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    split_idx = split_idx + self.nGMM
    local sigma_1_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    split_idx = split_idx + self.nGMM
    local sigma_2_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]
    split_idx = split_idx + self.nGMM
    local rho_t = input[{{},{split_idx,split_idx+self.nGMM-1}}]

    local sampleSize = (#input)[1]
    
    
    local gamma = self.mixdist:clone():cdiv(self.mixdist_sum:expand(sampleSize, self.nGMM)) 
    ----------

    local dl_hat_e_t = e_t:clone()
    dl_hat_e_t:mul(-1)
    
    dl_hat_e_t:add(x3)
    
    local dl_hat_pi_t = pi_t - gamma
    
    local c = self.inv_1mrho2:clone()
    
    local c_sigma1 = torch.cmul(c, self.inv_sigma1)
    local x1_mu1_sigma1 = torch.cmul(self.mu_1_x_1, self.inv_sigma1)
    local cor_x_2_mu2_sigma2 = torch.cmul(self.mu_2_x_2, rho_t)
    cor_x_2_mu2_sigma2:cmul(self.inv_sigma2)
    local dl_hat_mu_1_t = torch.cmul(x1_mu1_sigma1 - cor_x_2_mu2_sigma2, c_sigma1)
    dl_hat_mu_1_t:cmul(-gamma)
    
    local c_sigma2 = torch.cmul(c, self.inv_sigma2)
    local x2_mu2_sigma2 = torch.cmul(self.mu_2_x_2, self.inv_sigma2)
    local cor_x_1_mu1_sigma1 = torch.cmul(self.mu_1_x_1, rho_t)
    cor_x_1_mu1_sigma1:cmul(self.inv_sigma1)
    local dl_hat_mu_2_t = torch.cmul(x2_mu2_sigma2 - cor_x_1_mu1_sigma1, c_sigma2)
    dl_hat_mu_2_t:cmul(-gamma)
    
    local dl_hat_sigma_1_t = torch.cmul(c, self.mu_1_x_1)
    dl_hat_sigma_1_t:cmul(self.inv_sigma1)
    dl_hat_sigma_1_t:cmul(x1_mu1_sigma1 - cor_x_2_mu2_sigma2)
    dl_hat_sigma_1_t:add(-1)
    dl_hat_sigma_1_t:cmul(-gamma)
    
    local dl_hat_sigma_2_t = torch.cmul(c, self.mu_2_x_2)
    dl_hat_sigma_2_t:cmul(self.inv_sigma2)
    dl_hat_sigma_2_t:cmul(x2_mu2_sigma2 - cor_x_1_mu1_sigma1)
    dl_hat_sigma_2_t:add(-1)
    dl_hat_sigma_2_t:cmul(-gamma)
    
    local dl_hat_rho_t = torch.cmul(self.mu_1_x_1, self.mu_2_x_2)
    dl_hat_rho_t:cmul(self.inv_sigma1)
    dl_hat_rho_t:cmul(self.inv_sigma2)
    local cz = torch.cmul(c, self.z)
    local rho_cz = torch.cmul(rho_t, (-cz) + 1)
    local dl_hat_rho_t = dl_hat_rho_t + rho_cz
    dl_hat_rho_t:cmul(-gamma)

    local concat = nn.JoinTable(2)
    if self.useGPU == true then
        concat:cuda()
    end
    local grad_input = concat:forward({dl_hat_e_t, dl_hat_pi_t, dl_hat_mu_1_t, dl_hat_mu_2_t, dl_hat_sigma_1_t, dl_hat_sigma_2_t, dl_hat_rho_t})


    self.gradInput = grad_input
    self.gradInput:cmul(self.mask:reshape(self.mask:size(1),1):expand(self.gradInput:size()))
    
    if self.sizeAverage then
        self.gradInput:div(self.gradInput:size(1))
    end
    return self.gradInput
end
