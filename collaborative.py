import numpy as np
#load r
r_raw =[]
f = open('/home/hemant/Desktop/cs606-lab8-reccomenders-girihemant19/recolab-data-master/collaborative/ratings.txt', 'r')
for line in f.readlines():
	r_raw.append(map(int, line.split(' ')))
r = np.matrix(r_raw)
#print(r)
#derive p and q
m = r.shape[0] # of users
n = r.shape[1] # of movies
p = []
q = [0]*n
print(m)
#print(r.A[14])
for row in r.A:
	user_watch = list(row).count(1)
	p.append(user_watch)
	for i in range(n):
		q[i] += row[i]
def movies(i):
	f = open('/home/hemant/Desktop/cs606-lab8-reccomenders-girihemant19/recolab-data-master/collaborative/items.txt', 'r')
	return f.readlines()[i].strip('\n')

def diag_power(my_list):
        x=[1/np.sqrt(x) for x in my_list ]
	return np.diag(x)
#of items user i likes
p_minushalf = diag_power(p)
#print(p_minushalf)
#users liked item i
q_minushalf = diag_power(q)
#print(q_minushalf)

user_user = p_minushalf.dot(r).dot(np.transpose(r)).dot(p_minushalf).dot(r)
item_item = r.dot(q_minushalf).dot(np.transpose(r)).dot(r).dot(q_minushalf)
#print(sorted(user_user[499][:100])[:5])
#filtering recommendations for the 500 th user of the system.(Mr. A)
print('\nuser-user collaborative filtering results\n')
first_100 = list(user_user.A[499])[:100]
top_5 = sorted(first_100, reverse=True)[:5]
for score in top_5:
	print(movies(first_100.index(score))+' -> '+str(score))
#filtering recommendations for the 500 th user of the system.(Mr. A)
print('\nitem-item collaborative filtering results\n')
first_100 = list(item_item.A[499])[:100]
top_5 = sorted(first_100, reverse=True)[:5]
for score in top_5:
	print(movies(first_100.index(score))+' -> '+str(score))
