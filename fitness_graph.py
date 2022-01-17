from plotnine import ggplot, aes, geom_line
import csv
import matplotlib.pyplot as plt
fits = []
with open('stats.csv') as file:
    fitness = list(csv.reader(file))


for fit in fitness:
    fits.append([float(fit[0])])


print(fits)
plt.plot(fits, label="Fits")
plt.title("Average Fit Over Generations")
plt.ylabel("Fit")
plt.xlabel("Generations")
plt.show()

