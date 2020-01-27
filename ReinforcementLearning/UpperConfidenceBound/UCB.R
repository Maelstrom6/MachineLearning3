
dataset = read.csv("Ads_CTR_Optimisation.csv")

N = 10000  # number of rounds to play
d = 10  # number of different ads
n_i = integer(d)  # number of times ad i was selected up to round n
r_i = integer(d)  # sum of rewards of ad i up to round n
ads_selected= integer(0)

for (n in 1:N) {
  max_index = 0
  max_bound = 0
  
  for (i in 1:d) {
    if (n_i[i] == 0){
      r_bar_i = 0.5
      delta_i = 10000
    }else{
      r_bar_i = (r_i[i] / n_i[i])  # the sample proportion of rewards
      delta_i = (sqrt(3 / 2 * log(n) / n_i[i]))  # the sample variance of rewards
    }
    upper_bound = r_bar_i + delta_i
    if (max_bound < upper_bound){
      max_bound = upper_bound
      max_index = i
    }
  }
  
  result = dataset[n, max_index]
  n_i[max_index] = n_i[max_index] + 1
  r_i[max_index] = r_i[max_index] + result
  ads_selected = append(ads_selected, max_index)
}

print(n_i)
print(r_i)
print(sum(r_i))

hist(ads_selected,
     col = "blue", 
     main = paste("Histogram of ad selections"), 
     xlab = "Add number", 
     ylab = "Number of selections")

plot(1:d, r_i/n_i)
