
dataset = read.csv("Ads_CTR_Optimisation.csv")

N = 10000  # number of rounds to play
d = 10  # number of different ads
r0_i = integer(d)  # sum of punishments of ad i up to round n
r1_i = integer(d)  # sum of rewards of ad i up to round n
ads_selected= integer(0)

for (n in 1:N) {
  max_index = 0
  max_random = 0
  
  for (i in 1:d) {
    random_beta = rbeta(n = 1,
                        shape1 = r1_i[i] + 1, 
                        shape2 = r0_i[i] + 1)
    if (max_random < random_beta){
      max_random = random_beta
      max_index = i
    }
  }
  
  result = dataset[n, max_index]
  r0_i[max_index] = r0_i[max_index] + 1 - result
  r1_i[max_index] = r1_i[max_index] + result
  ads_selected = append(ads_selected, max_index)
}

print(sum(r1_i))

hist(ads_selected,
     col = "blue", 
     main = paste("Histogram of ad selections"), 
     xlab = "Add number", 
     ylab = "Number of selections")

plot(1:d, r_i/n_i)
