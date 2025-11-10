import numpy as np
import pandas
import math
import threading

long_length = 9
short_length = 2
batch_size = 5
total_num_of_requests = 20
short_in_short_out_percentage = 0.1
short_in_long_out_percentage = 0.45
long_in_short_out_percentage = 0.45
long_in_long_out_percentage = 0.1
itr_diff_tolerance_for_reassignment = 5

with open('request_distribution.txt', 'r') as reqDistribution:
    for eachLine in reqDistribution:
        parameters = eachLine.strip().split(" ")
        total_num_of_requests = float(parameters[0])
        short_in_short_out_percentage = float(parameters[1])
        short_in_long_out_percentage = float(parameters[2])
        long_in_short_out_percentage = float(parameters[3])
        long_in_long_out_percentage = float(parameters[4])
        break
    
# print(total_num_of_requests, short_in_short_out_percentage, short_in_long_out_percentage, long_in_short_out_percentage, long_in_long_out_percentage)

short_in_short_out_totReq = int(math.ceil(total_num_of_requests*short_in_short_out_percentage))
short_in_long_out_totReq = int(math.ceil(total_num_of_requests*short_in_long_out_percentage))
long_in_short_out_totReq = int(math.ceil(total_num_of_requests*long_in_short_out_percentage))
long_in_long_out_totReq = int(math.ceil(total_num_of_requests*long_in_long_out_percentage))

req_id = 0
short_in_short_out_whichReqs = []
short_in_long_out_whichReqs = []
long_in_short_out_whichReqs = []
long_in_long_out_whichReqs = []
empty_slots = 0
total_iterations = 0

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
gpuIterList = []
no_of_gpus = 2

for i in range(0, no_of_gpus):
    gpuIterList.append([])
# print(gpuIterList)

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

prefillEndIter = []
for i in range (0, actual_total_num_of_requests):
    if (i in short_in_short_out_whichReqs) or (i in long_in_short_out_whichReqs):
        decode_remaining[i] = short_length
        actual_decodes[i] = short_length
    else:
        decode_remaining[i] = long_length
        actual_decodes[i] = long_length
    prefillEndIter.append(0)

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

#TO DO: change to generalizable.
GPU_prefill_reqIds = []
GPU_decode_reqIds = []
for i in range(0, no_of_gpus):
    GPU_prefill_reqIds.append([])
    GPU_decode_reqIds.append([])
# GPU_prefill_reqIds[0] = []
# GPU_decode_reqIds[0] = []
# GPU_prefill_reqIds[1] = []
# GPU_decode_reqIds[1] = []

# GPU0: long input along with short output.
# GPU1: short input along with long output.


for reqId in short_in_short_out_whichReqs:
    GPU_decode_reqIds[1].append(reqId)
    GPU_prefill_reqIds[1].append(reqId)
for reqId in short_in_long_out_whichReqs:
    GPU_prefill_reqIds[1].append(reqId)
    GPU_decode_reqIds[1].append(reqId)
for reqId in long_in_short_out_whichReqs:
    GPU_prefill_reqIds[0].append(reqId)
    GPU_decode_reqIds[0].append(reqId)
for reqId in long_in_long_out_whichReqs:
    GPU_prefill_reqIds[0].append(reqId)
    GPU_decode_reqIds[1].append(reqId)
    


short_in_short_out_mid = short_in_short_out_totReq // 2
short_in_long_out_mid = short_in_long_out_totReq // 2
long_in_short_out_mid = long_in_short_out_totReq // 2
long_in_long_out_mid = long_in_long_out_totReq // 2

print("GPU0_prefills:",GPU_prefill_reqIds[0])
print("GPU0_decodes:",GPU_decode_reqIds[0])
print("GPU1_prefills:",GPU_prefill_reqIds[1])
print("GPU1_decodes:",GPU_decode_reqIds[1])
# print(prefill_remaining)
# print(decode_remaining)



def allDone_check(GPU_id):
    is_remaining_ours = 0
    if GPU_id == 0:
        for reqId in GPU_prefill_reqIds[0]:
            if (prefill_remaining[reqId] > 0):
                is_remaining_ours = 1
                break
        for reqId in GPU_decode_reqIds[0]:
            if (decode_remaining[reqId] > 0):
                is_remaining_ours = 1
                break
    else:
        for reqId in GPU_prefill_reqIds[1]:
            if (prefill_remaining[reqId] > 0):
                is_remaining_ours = 1
                break
        for reqId in GPU_decode_reqIds[1]:
            if (decode_remaining[reqId] > 0):
                is_remaining_ours = 1
                break
    return is_remaining_ours

def GPU0_processing():
    global empty_slots
    global total_iterations, maxIter, minIter
    global gpuIterList
    print("Ours GPU0 Processing:")
    # print("Batch size:", batch_size)
    gpu_id = 0
    iter = 1
    while True:
        # break
        if allDone_check(gpu_id) == 0:
            break 
        # print("GPU0: Iteration "+str(iter)+": ",end="")
        currItrProcessed = []
        currItrTokenList = []
        for batchSlot in range(0, batch_size):
            batchSlotDone = 0
            for reqId in GPU_decode_reqIds[0]:
                if reqId in currItrProcessed:
                    continue
                if (prefill_remaining[reqId] == 0) and (decode_remaining[reqId] > 0):
                    if iter <= prefillEndIter[reqId]:
                        continue
                    # print("D"+str(reqId)+" ",end="")
                    strToAdd = "D"+str(reqId)
                    # gpuIterList[gpu_id]
                    currItrTokenList.append(strToAdd)
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
                for reqId in GPU_prefill_reqIds[0]:
                    # if reqId in currItrProcessed:
                        # continue
                    if (prefill_remaining[reqId] > 0):
                        # print("P"+str(reqId)+" ",end="")
                        strToAdd = "P"+str(reqId)
                        currItrTokenList.append(strToAdd)
                        batchSlotDone = 1
                        prefill_remaining[reqId] = prefill_remaining[reqId] - 1
                        currItrProcessed.append(reqId)
                        
                        # if reqId in long_in_long_out_whichReqs:
                        prefillEndIter[reqId] = iter
                        break
            if batchSlotDone == 0:
                empty_slots = empty_slots + 1
        # print("")
        gpuIterList[gpu_id].append(currItrTokenList)
        iter = iter + 1
    total_iterations = total_iterations + (iter - 1)
    maxIter = max(maxIter, (iter-1))
    minIter = min(minIter, (iter-1))

# print("")

def GPU1_processing():
    global empty_slots
    global total_iterations, maxIter, minIter
    global gpuIterList
    gpu_id = 1
    print("Ours GPU1 Processing:")
    iter = 1
    while True:
        # break
        if allDone_check(gpu_id) == 0:
            break 
        # print("GPU1: Iteration "+str(iter)+": ",end="")
        currItrProcessed = []
        currItrTokenList = []
        for batchSlot in range(0, batch_size):
            batchSlotDone = 0
            for reqId in GPU_decode_reqIds[1]:
                if reqId in currItrProcessed:
                    continue
                if (prefill_remaining[reqId] == 0) and (decode_remaining[reqId] > 0):
                    # if reqId in long_in_long_out_whichReqs:
                    if iter <= prefillEndIter[reqId]:
                        continue
                    # print("D"+str(reqId)+" ",end="")
                    strToAdd = "D"+str(reqId)
                    currItrTokenList.append(strToAdd)
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
                for reqId in GPU_prefill_reqIds[1]:
                    # if reqId in currItrProcessed:
                        # continue
                    if (prefill_remaining[reqId] > 0):
                        # print("P"+str(reqId)+" ",end="")
                        strToAdd = "P"+str(reqId)
                        currItrTokenList.append(strToAdd)
                        batchSlotDone = 1
                        prefill_remaining[reqId] = prefill_remaining[reqId] - 1
                        currItrProcessed.append(reqId)
                        prefillEndIter[reqId] = iter
                        break
            if batchSlotDone == 0:
                empty_slots = empty_slots + 1
        # print("")
        gpuIterList[gpu_id].append(currItrTokenList)
        iter = iter + 1
        
    total_iterations = total_iterations + (iter - 1)
    maxIter = max(maxIter, (iter-1))
    minIter = min(minIter, (iter-1))

def remove_empty_iterations(gpu_id):
    global gpuIterList
    empty_iter_indices = []
    runningIndex = 0
    for iterList in gpuIterList[gpu_id]:
        if len(iterList) == 0:
            empty_iter_indices.append(runningIndex)
        runningIndex += 1
    for index in empty_iter_indices:
        gpuIterList[gpu_id].pop(index)

# def reassign_a_req(req_id, src_gpuId, dest_gpuId, transfer_mode=2):
#     #transfer_mode = 0: only prefills, 1: only decodes, 2: prefills+decodes.
#     #clear from src_gpuId
#     global gpuIterList
#     for iterList in gpuIterList[src_gpuId]:
#         # tokenCnt = 0
#         tokensToBeDeletedList = []
#         for token in iterList:
#             token_reqId = int(token[1:])
#             if token_reqId == req_id:
#                 tokensToBeDeletedList.append(token)
#         for token in tokensToBeDeletedList:
#             iterList.remove(token)

#     #remove the empty iterations after the clearing step from the src gpu.
#     remove_empty_iterations(src_gpuId)

#     #migrate to dest_gpuId
#     #if prefill is also reassigned, then can fill up from beginning
#     if (transfer_mode == 0 or transfer_mode == 2): #prefill transferring stage.
        
#         for iterList in gpuIterList[dest_gpuId]:
#             how_many_empty_slots = batch_size - len(iterList)
#             for i in range(0, how_many_empty_slots):

#     #if only decode, then decode fill up can only happen after prefill of the req. is done.

#     #remove the empty iterations after the migrating step from the dest GPU.

def calcEmptySlots(gpu_id):
    global gpuIterList
    calc_empty_slots = 0
    for iterList in gpuIterList[gpu_id]:
        calc_empty_slots += max(0, batch_size - len(iterList))

    return calc_empty_slots

dirty_reqIds = []    
def findMaxTokenReq(gpu_id):
    global GPU_prefill_reqIds, GPU_decode_reqIds, dirty_reqIds
    maxTokenCnt = -1000000
    finalReqId = -1
    for reqId in GPU_prefill_reqIds[gpu_id]:
        if reqId in dirty_reqIds:
            continue
        tokenCnt = actual_prefills[reqId]
        if reqId in GPU_decode_reqIds[gpu_id]:
            tokenCnt += actual_decodes[reqId]
        if (tokenCnt > maxTokenCnt):
            maxTokenCnt = tokenCnt
            finalReqId = reqId 
    for reqId in GPU_decode_reqIds[gpu_id]:
        if reqId in dirty_reqIds:
            continue
        tokenCnt = actual_decodes[reqId]
        if reqId in GPU_prefill_reqIds[gpu_id]:
            tokenCnt += actual_prefills[reqId]
        if (tokenCnt > maxTokenCnt):
            maxTokenCnt = tokenCnt
            finalReqId = reqId

    return maxTokenCnt, finalReqId

def reassignment():
    #find mid point for both GPUs end.
    global gpuIterList, GPU_prefill_reqIds, GPU_decode_reqIds, dirty_reqIds
    midEnd = int(math.ceil((len(gpuIterList[0])+len(gpuIterList[1]))/2))

    src_gpuId = 0
    dest_gpuId = 1
    if (len(gpuIterList[0]) < len(gpuIterList[1])):
        src_gpuId = 1
        dest_gpuId = 0
    
    #find which reqs & what prefills/decodes or both need to be reassigned for each req.., only chose a req. if req. empty slots are close to Mid.
    #for each req chosen in above step:
        #checkpoint prev. gpuiterlists for all GPUs.
        #call reassign_a_req func.
        #if leads to worse maxIter, then revert to checkpoint.
        #update the prefills & decodes assigned to associated GPUs.
    emptySlotsInDestGPU = calcEmptySlots(dest_gpuId) + max(0, (midEnd - len(gpuIterList[dest_gpuId])))*batch_size
    while True:
        should_go_on = 0
        for reqId in GPU_prefill_reqIds[src_gpuId]:
            if reqId not in dirty_reqIds:
                should_go_on = 1
        for reqId in GPU_decode_reqIds[src_gpuId]:
            if reqId not in dirty_reqIds:
                should_go_on = 1
            
        if should_go_on == 0:
            break
        maxTokenCnt, maxTokenReqId = findMaxTokenReq(src_gpuId)

        if (emptySlotsInDestGPU >= maxTokenCnt):
            emptySlotsInDestGPU -= maxTokenCnt 
            if maxTokenReqId in GPU_prefill_reqIds[src_gpuId]:
                GPU_prefill_reqIds[src_gpuId].remove(maxTokenReqId)
                GPU_prefill_reqIds[dest_gpuId].append(maxTokenReqId)
            if maxTokenReqId in GPU_decode_reqIds[src_gpuId]:
                GPU_decode_reqIds[src_gpuId].remove(maxTokenReqId)
                GPU_decode_reqIds[dest_gpuId].append(maxTokenReqId)
        else:
            dirty_reqIds.append(maxTokenReqId)

def GPU_utilizationCalc():

    global gpuIterList
    filled_up_slots = 0
    for gpu_id in range(0, no_of_gpus):
        for iterList in gpuIterList[gpu_id]:
            filled_up_slots += len(iterList)
    maxIter = max(len(gpuIterList[0]), len(gpuIterList[1]))
    print("MaxIter:", maxIter)
    gpu_util = filled_up_slots/(maxIter*no_of_gpus*batch_size)
    gpu_util *= 100.0

    return gpu_util

def ttft_calc():
    global ttft_list
    avgTTFT = sum(ttft_list)/len(ttft_list)
    ttft_list_npArray = np.array(ttft_list)
    q_25 = np.quantile(ttft_list_npArray, 0.25)
    q_50 = np.quantile(ttft_list_npArray, 0.5)
    q_75 = np.quantile(ttft_list_npArray, 0.75)
    print("#########################")
    print("Percentiles:", q_25, q_50, q_75)
    return avgTTFT

def tbt_calc():
    global tbt_list
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
    return avgTBT

def completion_time_calc():
    global completion_times
    avgCompletionTime = sum(completion_times)/len(completion_times)

    # print("Completion times:", completion_times)
    return avgCompletionTime

if __name__=="__main__":
    # t1 = threading.Thread(target=GPU0_processing)
    # t2 = threading.Thread(target=GPU1_processing)

    # t1.start()
    # t2.start()

    # t1.join()
    # t2.join()
    print("")
    GPU0_processing()
    print("")
    GPU1_processing()

    print("")
    gpu_utilization = GPU_utilizationCalc()
    print("GPU utilization:", gpu_utilization)

    avg_ttft = ttft_calc()
    print("Average TTFT:", avg_ttft)

    avg_tbt = tbt_calc()
    print("Average TBT:", avg_tbt)

    compl_time = completion_time_calc()
    print("Average Completion Time:", compl_time)

    # reassign_a_req(10, 0, 1)
    if (abs(len(gpuIterList[0])-len(gpuIterList[1])) > itr_diff_tolerance_for_reassignment):
        reassignment() #call only when itr diff is significant?-done

    print("")
    print("After Reassignment:")
    print("GPU0_prefills:",GPU_prefill_reqIds[0])
    print("GPU0_decodes:",GPU_decode_reqIds[0])
    print("GPU1_prefills:",GPU_prefill_reqIds[1])
    print("GPU1_decodes:",GPU_decode_reqIds[1])

    for i in range(0, actual_total_num_of_requests):
        prefill_remaining[i] = actual_prefills[i]
        decode_remaining[i] = actual_decodes[i]
        ttft_list[i] = 0
        tbt_list[i] = 0
        last_decode_list[i] = 0
        completion_times[i] = 0
        prefillEndIter[i] = 0
    for i in range(0, no_of_gpus):
        gpuIterList[i].clear()
    #do the categorical scheduling (1st step) again after the change caused by reassignment .
    #recalculate the metrics, show it & also show what are the reqs ids of the prefills & decodes assigned to each GPU.
    print("")
    GPU0_processing()
    print("")
    GPU1_processing()

    print("")
    gpu_utilization = GPU_utilizationCalc()
    print("GPU utilization:", gpu_utilization)

    avg_ttft = ttft_calc()
    print("Average TTFT:", avg_ttft)

    avg_tbt = tbt_calc()
    print("Average TBT:", avg_tbt)

    compl_time = completion_time_calc()
    print("Average Completion Time:", compl_time)

    #unn:
    # print(gpuIterList[0])
    # print(gpuIterList[1])
    # print(gpuIterList[0])
    # print(gpuIterList[1])
    avgTTFT = sum(ttft_list)/len(ttft_list)
    # print("TTFT List:", ttft_list)
    # print("Average TTFT:", avgTTFT)

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
    # print("TBT List:", tbt_list)
    # print("Average TBT:", avgTBT)

    gpu_underUtilized = empty_slots/(total_iterations*batch_size)
    gpu_utilized = 1.0 - gpu_underUtilized
    # # print("GPU Utilization:", gpu_utilized*100.0)
    # # print("Completed!")

    empty_slots = empty_slots + (maxIter - minIter)*batch_size
    gpu_underUtilized = empty_slots/(maxIter*2*batch_size) #TO DO: if GPUs>2, then this 2 needs chaning.
    gpu_utilized = 1.0 - gpu_underUtilized
    # print("GPU Utilization:", gpu_utilized*100.0)

    # print("Completion Times:", completion_times)
    avgCompletionTime = sum(completion_times)/len(completion_times)
    # print("Average Completion Time:", avgCompletionTime)

# print(actual_total_num_of_requests)
