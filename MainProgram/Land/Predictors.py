stats = df.corr()[['meantempm']].sort_values('meantempm') 

elements = []

for j,ele in enumerate(stats.meantempm):
    if (abs(ele) > 0.5):
        elements.append(stats.index[j])
        print(stats.index[j])
        
predictors = elements