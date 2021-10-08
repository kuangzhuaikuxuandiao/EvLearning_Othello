import numpy as np
import copy
import random
import math

def hiddenfunction(x,y):
    return x**2 + 2*y + 3*x + 5

def buildhiddenset():
    X = []
    y = []
    for i in range(100):
        x_ = random.randint(0, 40)
        y_ = random.randint(0, 40)
        X.append([x_, y_])
        y.append(hiddenfunction(x_, y_))
    return np.array(X), np.array(y)

class fwrapper:
    def __init__(self,function,params,name):
        self.function = function
        self.childcount = params
        self.name = name

class node:
    def __init__(self,fw,children):
        self.function = fw.function
        self.name = fw.name
        self.children=children

    def evaluate(self, inp):
        return self.function([n.evaluate(inp) for n in self.children])

    def display(self,indent=0):
        print((' '*indent)+self.name)
        for c in self.children:
            c.display(indent+1)

class paramnode:
    def __init__(self, idx):
        self.idx = idx

    def evaluate(self, inp):
        return inp[self.idx]

    def display(self, indent=0):
        print('%sp%d' % (' ' * indent, self.idx))

class constnode:
    def __init__(self, v):
        self.v = v

    def evaluate(self, inp):
        return self.v

    def display(self, indent=0):
        print('%s%d' % (' ' * indent, self.v))


addw=fwrapper(lambda l:l[0]+l[1],2,'add')
subw=fwrapper(lambda l:l[0]-l[1],2,'subtract')
mulw=fwrapper(lambda l:l[0]*l[1],2,'multiply')

def iffunc(l):
    if l[0] > 0:
        return l[1]
    else:
        return l[2]

ifw=fwrapper(iffunc,3,'if')

def isgreater(l):
    if l[0] > l[1]:
        return 1
    else:
        return 0

gtw=fwrapper(isgreater,2,'isgreater')
flist=[addw,mulw,ifw,gtw,subw]  # function set
# terminal set

# initialization
def makerandomtree(pc,maxdepth=4,fpr=0.5,ppr=0.6):
    if random.random() < fpr and maxdepth > 0:
        f = random.choice(flist)
        children = [makerandomtree(pc,maxdepth-1,fpr,ppr) for i in range(f.childcount)]
        return node(f,children)
    elif random.random() < ppr:
        return paramnode(random.randint(0,pc-1))
    else:
        return constnode(random.randint(0,10))

def scorefunction(tree,s):
    dif = 0
    for data in s:
        v = tree.evaluate([data[0],data[1]])
        dif += abs(v-data[2])
    return dif

def mutation(t,pc,probchange=0.1):
    if random.random() < probchange:
        return makerandomtree(pc)
    else:
        result = copy.deepcopy(t)
        if hasattr(t, "children"):
            result.children = [mutation(c,pc,probchange) for c in t.children]
        return result

def crossover(t1,t2,probswap=0.7,top=1):
    if random.random() < probswap and not top:
        return copy.deepcopy(t2)
    else:
        result = copy.deepcopy(t1)
        if hasattr(t1,'children') and hasattr(t2,'children'):
            result.children = [crossover(c,random.choice(t2.children), probswap,0) for c in t1.children]
        return result


def getrankfunction(dataset):
    def rankfunction(population):
        scores = [(scorefunction(t, dataset), t) for t in population]
        scores.sort()
        return scores
    return rankfunction


def evolve(pc, popsize, rankfunction, maxgen=500,
           mutationrate=0.1, breedingrate=0.4, pexp=0.7, pnew=0.05):
    # Returns a random number, tending towards lower numbers. The lower pexp
    # is, more lower numbers you will get
    def selectindex():
        return int(math.log(random.random()) / math.log(pexp))

    # Create a random initial population
    scores = None
    population = [makerandomtree(pc) for i in range(popsize)]
    for i in range(maxgen):
        scores = rankfunction(population)
        print("function score: ", scores[0][0])
        if scores[0][0] == 0:
            break
        # The two best always make it
        newpop = [scores[0][1], scores[1][1]]

        # Build the next generation
        while len(newpop) < popsize:
            if random.random() > pnew:
                newpop.append(mutation(
                    crossover(scores[selectindex()][1],
                              scores[selectindex()][1],
                              probswap=breedingrate),
                    pc, probchange=mutationrate))
            else:
                # Add a random node to mix things up
                newpop.append(makerandomtree(pc))

        population = newpop
    scores[0][1].display()
    return scores[0][1]

