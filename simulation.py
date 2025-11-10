import numpy as np
import pandas
import math
import random
import time

from datetime import datetime


long_length = 9
short_length = 2
batch_size = 5
total_num_of_requests = 20
short_in_short_out_percentage = 0.1
short_in_long_out_percentage = 0.45
long_in_short_out_percentage = 0.45
long_in_long_out_percentage = 0.1

with open('request_distribution.txt', 'r') as reqDistribution:
    for eachLine in reqDistribution:
        parameters = eachLine.strip().split(" ")
        total_num_of_requests = float(parameters[0])
        short_in_short_out_percentage = float(parameters[1])
        short_in_long_out_percentage = float(parameters[2])
        long_in_short_out_percentage = float(parameters[3])
        long_in_long_out_percentage = float(parameters[4])
        break

short_in_short_out_totReq = int(math.ceil(total_num_of_requests*short_in_short_out_percentage))
short_in_long_out_totReq = int(math.ceil(total_num_of_requests*short_in_long_out_percentage))
long_in_short_out_totReq = int(math.ceil(total_num_of_requests*long_in_short_out_percentage))
long_in_long_out_totReq = int(math.ceil(total_num_of_requests*long_in_long_out_percentage))

req_id = 0
short_in_short_out_whichReqs = []
short_in_long_out_whichReqs = []
long_in_short_out_whichReqs = []
long_in_long_out_whichReqs = []

for i in range (0, short_in_short_out_totReq):
    short_in_short_out_whichReqs.append(req_id)
    req_id = req_id + 1
for i in range (0, short_in_long_out_totReq):
    short_in_long_out_whichReqs.append(req_id)
    req_id = req_id + 1
for i in range (0, long_in_short_out_totReq):
    long_in_short_out_whichReqs.append(req_id)
    req_id = req_id + 1
for i in range (0, long_in_long_out_totReq):
    long_in_long_out_whichReqs.append(req_id)
    req_id = req_id + 1

actual_total_num_of_requests = req_id

prefill_remaining = []
decode_remaining = []
ttft_list = []
tbt_list = []
last_decode_list = []
actual_prefills = []
actual_decodes = []
completion_times = []
empty_slots = 0
total_iterations = 0
maxIter = -1
minIter = 1000000000

for i in range(0, actual_total_num_of_requests):
    prefill_remaining.append(0)
    decode_remaining.append(0)
    ttft_list.append(0)
    tbt_list.append(0)
    last_decode_list.append(0)
    actual_prefills.append(0)
    actual_decodes.append(0)
    completion_times.append(0)

for i in range (0, actual_total_num_of_requests):
    if (i in short_in_short_out_whichReqs) or (i in short_in_long_out_whichReqs):
        prefill_remaining[i] = short_length
        actual_prefills[i] = short_length
    else:
        prefill_remaining[i] = long_length
        actual_prefills[i] = long_length

for i in range (0, actual_total_num_of_requests):
    if (i in short_in_short_out_whichReqs) or (i in long_in_short_out_whichReqs):
        decode_remaining[i] = short_length
        actual_decodes[i] = short_length
    else:
        decode_remaining[i] = long_length
        actual_decodes[i] = long_length

fileLineCnt = 0
with open('request_distribution.txt', 'r') as reqDistribution:
    for eachLine in reqDistribution:
        if fileLineCnt > 0:
            parameters = eachLine.strip().split(" ")
            reqId = int(parameters[0])
            prefill_length = int(parameters[1])
            decode_length = int(parameters[2])
            actual_prefills[reqId] = prefill_length
            prefill_remaining[reqId] = prefill_length
            actual_decodes[reqId] = decode_length
            decode_remaining[reqId] = decode_length
        fileLineCnt += 1

for i in range (0, actual_total_num_of_requests):
    print("r"+str(i)+": ", end="")
    for j in range (0, actual_prefills[i]):
        print("P"+str(i)+" ",end="")
    for j in range (0, actual_decodes[i]):
        print("D"+str(i)+" ",end="")
    print("")

print("Short input short output percentage:", short_in_short_out_totReq/actual_total_num_of_requests, end="")
print("")
print("Short input long output percentage:", short_in_long_out_totReq/actual_total_num_of_requests, end="")
print("")
print("Long input short output percentage:", long_in_short_out_totReq/actual_total_num_of_requests, end="")
print("")
print("Long input long output percentage:", long_in_long_out_totReq/actual_total_num_of_requests, end="")
print("")

GPU0_reqIds = []
GPU1_reqIds = []

def uniform_workloadAware():
    short_in_short_out_mid = short_in_short_out_totReq // 2
    short_in_long_out_mid = short_in_long_out_totReq // 2
    long_in_short_out_mid = long_in_short_out_totReq // 2
    long_in_long_out_mid = long_in_long_out_totReq // 2

    for i in range (0, short_in_short_out_mid):
        GPU0_reqIds.append(short_in_short_out_whichReqs[i])
    for i in range (short_in_short_out_mid, short_in_short_out_totReq):
        GPU1_reqIds.append(short_in_short_out_whichReqs[i])

    for i in range (0, short_in_long_out_mid):
        GPU0_reqIds.append(short_in_long_out_whichReqs[i])
    for i in range (short_in_long_out_mid, short_in_long_out_totReq):
        GPU1_reqIds.append(short_in_long_out_whichReqs[i])

    for i in range (0, long_in_short_out_mid):
        GPU0_reqIds.append(long_in_short_out_whichReqs[i])
    for i in range (long_in_short_out_mid, long_in_short_out_totReq):
        GPU1_reqIds.append(long_in_short_out_whichReqs[i])

    for i in range (0, long_in_long_out_mid):
        GPU0_reqIds.append(long_in_long_out_whichReqs[i])
    for i in range (long_in_long_out_mid, long_in_long_out_totReq):
        GPU1_reqIds.append(long_in_long_out_whichReqs[i])

def randomDivision():
    how_many_reqs_eachGPU = actual_total_num_of_requests // 2
    random.seed(int(datetime.now().timestamp()))
    for i in range (0, how_many_reqs_eachGPU):
        while True:
            randomIndex = random.randint(0, actual_total_num_of_requests-1)
            if randomIndex not in GPU0_reqIds:
                GPU0_reqIds.append(randomIndex)
                break 

    for i in range (0, actual_total_num_of_requests):
        if i not in GPU0_reqIds:
            GPU1_reqIds.append(i)

randomDivision()
# print(GPU0_reqIds)
# print(prefill_remaining)
# uniform_workloadAware()
# print(decode_remaining)
print("GPU0 requests:", GPU0_reqIds)
print("GPU1 requests:", GPU1_reqIds)


def allDone_check(GPU_id):
    is_remaining = 0
    if GPU_id == 0:
        for i in range (0, len(GPU0_reqIds)):
            if (prefill_remaining[GPU0_reqIds[i]] > 0) or (decode_remaining[GPU0_reqIds[i]] > 0):
                is_remaining = 1
                break
    else:
        for i in range (0, len(GPU1_reqIds)):
            if (prefill_remaining[GPU1_reqIds[i]] > 0) or (decode_remaining[GPU1_reqIds[i]] > 0):
                is_remaining = 1
                break
    return is_remaining

print("")
print("Sarathi GPU0 Processing:")
iter = 1
while True:
    if allDone_check(0) == 0:
        break 
    print("GPU0: Iteration "+str(iter)+": ",end="")
    currItrProcessed = []
    for batchSlot in range(0, batch_size):
        batchSlotDone = 0
        for reqId in GPU0_reqIds:
            if reqId in currItrProcessed:
                continue
            if (prefill_remaining[reqId] == 0) and (decode_remaining[reqId] > 0):
                print("D"+str(reqId)+" ",end="")
                batchSlotDone = 1
                decode_remaining[reqId] = decode_remaining[reqId] - 1
                currItrProcessed.append(reqId)

                if (last_decode_list[reqId] == 0):
                    ttft_list[reqId] = iter 
                else:
                    # dummy = 1 #TO DO: input code for tbt calc.
                    tbt_list[reqId] = tbt_list[reqId] + (iter - last_decode_list[reqId])
                last_decode_list[reqId] = iter
                completion_times[reqId] = iter 
                break
        if batchSlotDone == 0:
            for reqId in GPU0_reqIds:
                # if reqId in currItrProcessed:
                    # continue
                if (prefill_remaining[reqId] > 0):
                    print("P"+str(reqId)+" ",end="")
                    batchSlotDone = 1
                    prefill_remaining[reqId] = prefill_remaining[reqId] - 1
                    currItrProcessed.append(reqId)
                    
                    break
        if batchSlotDone == 0:
            empty_slots = empty_slots + 1
    print("")
    iter = iter + 1
total_iterations = total_iterations + (iter - 1)
maxIter = max(maxIter, (iter-1))
minIter = min(minIter, (iter-1))
print("")

print("Sarathi GPU1 Processing:")
iter = 1
while True:
    if allDone_check(1) == 0:
        break 
    print("GPU1: Iteration "+str(iter)+": ",end="")
    currItrProcessed = []
    for batchSlot in range(0, batch_size):
        batchSlotDone = 0
        for reqId in GPU1_reqIds:
            if reqId in currItrProcessed:
                continue
            if (prefill_remaining[reqId] == 0) and (decode_remaining[reqId] > 0):
                print("D"+str(reqId)+" ",end="")
                batchSlotDone = 1
                decode_remaining[reqId] = decode_remaining[reqId] - 1
                currItrProcessed.append(reqId)


                if (last_decode_list[reqId] == 0):
                    ttft_list[reqId] = iter 
                else:
                    # dummy = 1 #TO DO: input code for tbt calc.
                    tbt_list[reqId] = tbt_list[reqId] + (iter - last_decode_list[reqId])
                
                last_decode_list[reqId] = iter
                completion_times[reqId] = iter 
                break
        if batchSlotDone == 0:
            for reqId in GPU1_reqIds:
                # if reqId in currItrProcessed:
                    # continue
                if (prefill_remaining[reqId] > 0):
                    print("P"+str(reqId)+" ",end="")
                    batchSlotDone = 1
                    prefill_remaining[reqId] = prefill_remaining[reqId] - 1
                    currItrProcessed.append(reqId)
                    break
        if batchSlotDone == 0:
            empty_slots = empty_slots + 1
    print("")
    iter = iter + 1

total_iterations = total_iterations + (iter - 1)
maxIter = max(maxIter, (iter-1))
minIter = min(minIter, (iter-1))

avgTTFT = sum(ttft_list)/len(ttft_list)
print("TTFT List:", ttft_list)
print("Average TTFT:", avgTTFT)

ttft_list_npArray = np.array(ttft_list)
q_25 = np.quantile(ttft_list_npArray, 0.25)
q_50 = np.quantile(ttft_list_npArray, 0.5)
q_75 = np.quantile(ttft_list_npArray, 0.75)
print("#########################")
print("Percentiles:", q_25, q_50, q_75)

# print("TBT List:", tbt_list)
for reqId in range (0, actual_total_num_of_requests):
    if (tbt_list[reqId] != 0):
        tbt_list[reqId] = tbt_list[reqId]/(actual_decodes[reqId]-1)
avgTBT = 0
sumTBT = 0
lenTBT = 0
for reqId in range (0, actual_total_num_of_requests):
    if (tbt_list[reqId] != 0):
        sumTBT = sumTBT + tbt_list[reqId]
        lenTBT = lenTBT + 1
avgTBT = sumTBT/lenTBT
print("TBT List:", tbt_list)
print("Average TBT:", avgTBT)

gpu_underUtilized = empty_slots/(total_iterations*batch_size)
gpu_utilized = 1.0 - gpu_underUtilized

empty_slots = empty_slots + (maxIter - minIter)*batch_size
gpu_underUtilized = empty_slots/(maxIter*2*batch_size) #TO DO: if GPUs>2, then this 2 needs chaning.
print("Max Iter:", maxIter)
gpu_utilized = 1.0 - gpu_underUtilized
print("GPU Utilization:", gpu_utilized*100.0)

print("Completion Times:", completion_times)
avgCompletionTime = sum(completion_times)/len(completion_times)
print("Average Completion Time:", avgCompletionTime)
# print(actual_total_num_of_requests)
# print ("hello world")