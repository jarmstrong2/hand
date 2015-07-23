require 'nn'

local YHat, parent = torch.class('nn.YHat', 'nn.Module')

function YHat:settarget(x)
    self.x1 = x[{{},{1}}]
    self.x2 = x[{{},{2}}]
    self.x3 = x[{{},{3}}]
end

function YHat:setmask(mask)
    self.mask = mask
end

function YHat:updateOutput(input)
    self.hat_e_t = input[{{},{1}}]
    self.hat_pi_t = input[{{},{2,21}}]
    self.hat_mu_1_t = input[{{},{22,41}}]
    self.hat_mu_2_t = input[{{},{42,61}}]
    self.hat_sigma_1_t = input[{{},{62,81}}]
    self.hat_sigma_2_t = input[{{},{82,101}}]
    self.hat_rho_t = input[{{},{102,121}}]

    self.e_t = (nn.Sigmoid()):forward(-self.hat_e_t)
    self.pi_t = (nn.SoftMax()):forward(self.hat_pi_t)
    self.mu_1_t = self.hat_mu_1_t:clone()
    self.mu_2_t =  self.hat_mu_2_t:clone()
    self.sigma_1_t = (nn.Exp()):forward(self.hat_sigma_1_t)
    self.sigma_2_t = (nn.Exp()):forward(self.hat_sigma_2_t)
    self.rho_t = (nn.Tanh()):forward(self.hat_rho_t)
    
    output = torch.cat(self.e_t:clone():double(), self.pi_t:clone():double(), 2)
    output = torch.cat(output, self.mu_1_t:clone():double(), 2)
    output = torch.cat(output, self.mu_2_t:clone():double(), 2)
    output = torch.cat(output, self.sigma_1_t:clone():double(), 2)
    output = torch.cat(output, self.sigma_2_t:clone():double(), 2)
    output = torch.cat(output, self.rho_t:clone():double(), 2)
    self.output = output
    
    return self.output
end

function YHat:updateGradInput(input, gradOutput)
    -- use gradOutout as mask

    self.hat_e_t = input[{{},{1}}]
    self.hat_pi_t = input[{{},{2,21}}]
    self.hat_mu_1_t = input[{{},{22,41}}]
    self.hat_mu_2_t = input[{{},{42,61}}]
    self.hat_sigma_1_t = input[{{},{62,81}}]
    self.hat_sigma_2_t = input[{{},{82,101}}]
    self.hat_rho_t = input[{{},{102,121}}]

    self.e_t = (nn.Sigmoid()):forward(-self.hat_e_t)
    self.pi_t = (nn.SoftMax()):forward(self.hat_pi_t)
    self.mu_1_t = self.hat_mu_1_t:clone()
    self.mu_2_t =  self.hat_mu_2_t:clone()
    self.sigma_1_t = (nn.Exp()):forward(self.hat_sigma_1_t)
    self.sigma_2_t = (nn.Exp()):forward(self.hat_sigma_2_t)
    self.rho_t = (nn.Tanh()):forward(self.hat_rho_t)
    
    sampleSize = (#input)[1]
    
    --responsibilities will separate calculation into gamma1 and gamma2
    
    inv_sigma1 = torch.pow(self.sigma_1_t + (10^-15), -1)
    inv_sigma2 = torch.pow(self.sigma_2_t + (10^-15), -1)
    
    gamma1 = torch.cmul(inv_sigma1, inv_sigma2)
    gamma1:cmul(torch.pow((-(torch.pow(self.rho_t, 2)) + 1 + (10^-15)), -0.5))
    gamma1:mul(1/(2*math.pi))
    
    mu_1_x_1 = (self.mu_1_t:clone()):mul(-1)
    mu_2_x_2 = (self.mu_2_t:clone()):mul(-1)
    
    x1_val = self.x1:expand(sampleSize, 20)
    x2_val = self.x2:expand(sampleSize, 20) 
    mu_1_x_1:add(x1_val)
    mu_2_x_2:add(x2_val) 
    
    gamma2_z_1 = torch.cmul(torch.pow(inv_sigma1, 2), torch.pow(mu_1_x_1,2))  
    gamma2_z_2 = torch.cmul(torch.pow(inv_sigma2, 2), torch.pow(mu_2_x_2,2))
    
    gamma2_z_3 = torch.cmul(inv_sigma1, inv_sigma2)
    gamma2_z_3:cmul(mu_1_x_1)
    gamma2_z_3:cmul(mu_2_x_2)
    
    gamma2_z_3:cmul(torch.mul(self.rho_t, 2))
    z = gamma2_z_1 + gamma2_z_2 - gamma2_z_3
    
    gamma2 = z:clone()
    gamma2:mul(-1)
    gamma2:cmul(torch.pow((-(torch.pow(self.rho_t, 2)) + 1 + (10^-15)):mul(2), -1))
    gamma2:exp()
    gamma = torch.cmul(gamma1, gamma2)
    gamma:cmul(self.pi_t)
    gamma_sum = torch.sum(gamma, 2)
    gamma_sum:add(10^-15)

    gamma_sum_val = gamma_sum:expand(sampleSize, 20)
    gamma:cmul(torch.pow(gamma_sum_val, -1))

    dl_hat_e_t = self.e_t:clone()
    dl_hat_e_t:mul(-1)
    
    dl_hat_e_t:add(self.x3)
    
    dl_hat_pi_t = self.pi_t - gamma
    
    c = torch.pow((-torch.pow(self.rho_t, 2)):add(1 + 10^-15), -1)
    
    c_sigma1 = torch.cmul(c, inv_sigma1)
    x1_mu1_sigma1 = torch.cmul(mu_1_x_1, inv_sigma1)
    cor_x_2_mu2_sigma2 = torch.cmul(mu_2_x_2, self.rho_t)
    cor_x_2_mu2_sigma2:cmul(inv_sigma2)
    dl_hat_mu_1_t = torch.cmul(x1_mu1_sigma1 - cor_x_2_mu2_sigma2, c_sigma1)
    dl_hat_mu_1_t:cmul(-gamma)
    
    c_sigma2 = torch.cmul(c, inv_sigma2)
    x2_mu2_sigma2 = torch.cmul(mu_2_x_2, inv_sigma2)
    cor_x_1_mu1_sigma1 = torch.cmul(mu_1_x_1, self.rho_t)
    cor_x_1_mu1_sigma1:cmul(inv_sigma1)
    dl_hat_mu_2_t = torch.cmul(x2_mu2_sigma2 - cor_x_1_mu1_sigma1, c_sigma2)
    dl_hat_mu_2_t:cmul(-gamma)
    
    dl_hat_sigma_1_t = torch.cmul(c, mu_1_x_1)
    dl_hat_sigma_1_t:cmul(inv_sigma1)
    dl_hat_sigma_1_t:cmul(x1_mu1_sigma1 - cor_x_2_mu2_sigma2)
    dl_hat_sigma_1_t:add(-1)
    dl_hat_sigma_1_t:cmul(-gamma)
    
    dl_hat_sigma_2_t = torch.cmul(c, mu_2_x_2)
    dl_hat_sigma_2_t:cmul(inv_sigma2)
    dl_hat_sigma_2_t:cmul(x2_mu2_sigma2 - cor_x_1_mu1_sigma1)
    dl_hat_sigma_2_t:add(-1)
    dl_hat_sigma_2_t:cmul(-gamma)
    
    dl_hat_rho_t = torch.cmul(mu_1_x_1, mu_2_x_2)
    dl_hat_rho_t:cmul(inv_sigma1)
    dl_hat_rho_t:cmul(inv_sigma2)
    cz = torch.cmul(c, z)
    rho_cz = torch.cmul(self.rho_t, (-cz) + 1)
    dl_hat_rho_t = dl_hat_rho_t + rho_cz
    dl_hat_rho_t:cmul(-gamma)
    
    output = torch.cat(dl_hat_e_t:double(), dl_hat_pi_t:double())
    output = torch.cat(output, dl_hat_mu_1_t:double())
    output = torch.cat(output, dl_hat_mu_2_t:double())
    output = torch.cat(output, dl_hat_sigma_1_t:double())
    output = torch.cat(output, dl_hat_sigma_2_t:double())
    output = torch.cat(output, dl_hat_rho_t:double())
    
    self.gradInput = output
    self.gradInput:cmul(self.mask)
    
    return self.gradInput
end
