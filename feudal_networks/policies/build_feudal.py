from feudal_policy import FeudalPolicy

class config():
    alpha = .5

if __name__ == '__main__':
    feudal = FeudalPolicy((80,80,3),(4,),config)
