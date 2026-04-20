python main.py -data MHEALTH -m CNN -algo FedMul -gr 40 -did 0 -am none -dm none
python main.py -data FashionMNIST -m CNN -algo FedMul -gr 40 -did 0 -am none -dm none
python main.py -data HAR -m CNN -algo FedMul -gr 40 -did 0 -am none -dm none
python main.py -data MHEALTH -m CNN -algo FedMul -gr 40 -did 0 -am badnets -dm none
python main.py -data FashionMNIST -m CNN -algo FedMul -gr 40 -did 0 -am badnets -dm none
python main.py -data HAR -m CNN -algo FedMul -gr 40 -did 0 -am badnets -dm none
python main.py -data MHEALTH -m CNN -algo FedMul -gr 40 -did 0 -am dba -dm none
python main.py -data FashionMNIST -m CNN -algo FedMul -gr 40 -did 0 -am dba -dm none
python main.py -data HAR -m CNN -algo FedMul -gr 40 -did 0 -am dba -dm none
python main.py -data MHEALTH -m CNN -algo FedMul -gr 40 -did 0 -am neurotoxin -dm none
python main.py -data FashionMNIST -m CNN -algo FedMul -gr 40 -did 0 -am neurotoxin -dm none
python main.py -data HAR -m CNN -algo FedMul -gr 40 -did 0 -am neurotoxin -dm none
python main.py -data MHEALTH -m CNN -algo FedMul -gr 40 -did 0 -am model_replacement -dm none
python main.py -data FashionMNIST -m CNN -algo FedMul -gr 40 -did 0 -am model_replacement -dm none
python main.py -data HAR -m CNN -algo FedMul -gr 40 -did 0 -am model_replacement -dm none
python main.py -data MHEALTH -m CNN -algo FedAvg -gr 40 -did 0 -am none
python main.py -data MHEALTH -m CNN -algo FedAvg -gr 40 -did 0 -am badnets
python main.py -data MHEALTH -m CNN -algo FedAvg -gr 40 -did 0 -am dba
python main.py -data MHEALTH -m CNN -algo FedAvg -gr 40 -did 0 -am neurotoxin
python main.py -data MHEALTH -m CNN -algo FedAvg -gr 40 -did 0 -am model_replacement
python main.py -data MHEALTH -m CNN -algo FedAS -gr 40 -did 0 -am none
python main.py -data MHEALTH -m CNN -algo FedAS -gr 40 -did 0 -am badnets 
python main.py -data MHEALTH -m CNN -algo FedAS -gr 40 -did 0 -am dba
python main.py -data MHEALTH -m CNN -algo FedAS -gr 40 -did 0 -am neurotoxin
python main.py -data MHEALTH -m CNN -algo FedAS -gr 40 -did 0 -am model_replacement