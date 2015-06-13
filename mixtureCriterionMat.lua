require 'nn'
local MixtureCriterion, parent = torch.class('nn.MixtureCriterion', 'nn.Criterion')

function MixtureCriterion:setmask(mask)
   self.mask = mask 
end

function MixtureCriterion:updateOutput(input, target)

    x1 = target[{{},{1}}]
    x2 = target[{{},{2}}]
    x3 = target[{{},{3}}]
    
    e_t = input[{{},{1}}]
    pi_t = input[{{},{2,21}}]
    mu_1_t = input[{{},{22,41}}]
    mu_2_t = input[{{},{42,61}}]
    sigma_1_t = input[{{},{62,81}}]
    sigma_2_t = input[{{},{82,101}}]
    rho_t = input[{{},{102,121}}]
    
    sampleSize = (#input)[1]
    
    inv_sigma1 = torch.pow(sigma_1_t + 10^-15, -1)
    inv_sigma2 = torch.pow(sigma_2_t + 10^-15, -1)
    
    mixdist1 = torch.cmul(inv_sigma1, inv_sigma2)
    mixdist1:cmul(torch.pow((-(torch.pow(rho_t, 2)) + 1), -0.5))
    mixdist1:mul(1/(2*math.pi))
    
    mu_1_x_1 = (mu_1_t:clone()):mul(-1)
    mu_2_x_2 = (mu_2_t:clone()):mul(-1)
    
    x1_val = x1:expand(sampleSize, 20)
    x2_val = x2:expand(sampleSize, 20) 
    mu_1_x_1:add(x1_val)
    mu_2_x_2:add(x2_val) 
    
    mixdist2_z_1 = torch.cmul(torch.pow(inv_sigma1, 2), torch.pow(mu_1_x_1,2))  
    mixdist2_z_2 = torch.cmul(torch.pow(inv_sigma2, 2), torch.pow(mu_2_x_2,2)) 
    mixdist2_z_3 = torch.cmul(inv_sigma1, inv_sigma2)
    
    mixdist2_z_3:cmul(mu_1_x_1)
    mixdist2_z_3:cmul(mu_2_x_2)
    
    mixdist2_z_3:cmul(torch.mul(rho_t, 2))
    z = mixdist2_z_1 + mixdist2_z_2 - mixdist2_z_3
    mixdist2 = z:clone()
    mixdist2:mul(-1)
    mixdist2:cmul(torch.pow((-(torch.pow(rho_t, 2)) + 1):mul(2), -1))
    mixdist2:exp()
    mixdist = torch.cmul(mixdist1, mixdist2)
    mixdist:cmul(pi_t)
    
    mixdist_sum = torch.sum(mixdist, 2)
    
    log_mixdist_sum = torch.log(mixdist_sum)
    
    log_e_t = e_t:clone()
    
    eq1 = torch.eq(x3, torch.ones(sampleSize, 1):cuda())
    eq1 = eq1:cuda()
    eq1:cmul(torch.log(e_t))
    neq1 = torch.ne(x3, torch.ones(sampleSize, 1):cuda())
    neq1 = neq1:cuda()
    neq1:cmul(torch.log(-e_t + 1))
    log_e_t = eq1 + neq1
    
    result = log_mixdist_sum + log_e_t
    result:mul(-1)
    result:cmul(self.mask)
    result = result:sum()
    
    return result
end

function MixtureCriterion:updateGradInput(input, target)
   return nil
end
