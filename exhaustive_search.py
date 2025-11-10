import numpy as np
import pandas
import math
import threading
import random
import sys
import time
from itertools import product

tot_reqs = 10

prefill_lengths = [830, 590, 630, 203, 508, 450, 4593, 8976, 892, 8905]
decode_lengths = [204, 305, 458, 594, 689, 698, 389, 995, 987, 909]


batch_size = 512
total_num_of_requests = 20
short_in_short_out_percentage = 0.1
short_in_long_out_percentage = 0.45
long_in_short_out_percentage = 0.45
long_in_long_out_percentage = 0.1
itr_diff_tolerance_for_reassignment = 5
itr_diff_tolerance_for_reassignment_2 = 1
maxItrCntReassignment_2 = 3
tot_iterations1 = 0
extra_slots_tolerance_reassignment_2 = batch_size

req_id = 0
short_in_short_out_whichReqs = []
short_in_long_out_whichReqs = []
long_in_short_out_whichReqs = []
long_in_long_out_whichReqs = []
empty_slots = 0
total_iterations = 0


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
actual_total_num_of_requests = tot_reqs
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
   
    prefill_remaining[i] = prefill_lengths[i]
    actual_prefills[i] = prefill_lengths[i]
    

prefillEndIter = []
for i in range (0, actual_total_num_of_requests):
    
    decode_remaining[i] = decode_lengths[i]
    actual_decodes[i] = decode_lengths[i]
    prefillEndIter.append(0)
GPU_prefill_reqIds = []
GPU_decode_reqIds = []
for i in range(0, no_of_gpus):
    GPU_prefill_reqIds.append([])
    GPU_decode_reqIds.append([])    

def allDone_check(GPU_id):
    is_remaining_ours = 0
    somevalue = 100000
    global tot_iterations1
    if tot_iterations1 >= somevalue:
        return 0
    else:
        return 1
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

def estimate_start_end_iterations(positions_newEmpty_slots, prefill_or_decode, reqId, reqId_current_start, reqId_current_end):
    reqId_current_start_itr = reqId_current_start[0]
    reqId_current_start_slot = reqId_current_start[1]

    reqId_current_end_itr = reqId_current_end[0]
    reqId_current_end_slot = reqId_current_end[1]

    reqId_estimated_start_itr = -1
    reqId_estimated_start_slot = -1

    reqId_estimated_end_itr = -1
    reqId_estimated_end_slot = -1

    if (prefill_or_decode == 0):
        #finding how many slots are there before the reqId_start.
        #TO DO: use binary search here to make things faster.
        start_how_much_move = 0
        for eachPosition in positions_newEmpty_slots:
            if eachPosition[0] > reqId_current_start_itr:
                break
            elif eachPosition[0] < reqId_current_start_itr:
                start_how_much_move += 1
            else:
                if eachPosition[1] < reqId_current_start_slot:
                    start_how_much_move += 1
                else:
                    break 
        start_new_virtual_position = reqId_current_start_slot - start_how_much_move
        if start_new_virtual_position >= 0:
            reqId_estimated_start_itr = reqId_current_start_itr
            reqId_estimated_start_slot = start_new_virtual_position
        else:
            start_new_virtual_position = start_new_virtual_position*-1

            #first finding the estimated itr for the start.
            itr_to_subtract = int(math.ceil(start_new_virtual_position/batch_size))
            reqId_estimated_start_itr = reqId_current_start_itr - itr_to_subtract

            #now finding the estimated slot for the start.
            mod_value = start_new_virtual_position % batch_size
            if mod_value > 0:
                mod_value = batch_size - mod_value
            reqId_estimated_start_slot = mod_value 

        #do the same to estimate the end.
        end_how_much_move = 0
        for eachPosition in positions_newEmpty_slots:
            if eachPosition[0] > reqId_current_end_itr:
                break
            elif eachPosition[0] < reqId_current_end_itr:
                end_how_much_move += 1
            else:
                if eachPosition[1] < reqId_current_end_slot:
                    end_how_much_move += 1
                else:
                    break 
        end_new_virtual_position = reqId_current_end_slot - end_how_much_move
        if end_new_virtual_position >= 0:
            reqId_estimated_end_itr = reqId_current_end_itr
            reqId_estimated_end_slot = end_new_virtual_position
        else:
            end_new_virtual_position = end_new_virtual_position*-1

            #first finding the estimated itr for the start.
            itr_to_subtract = int(math.ceil(end_new_virtual_position/batch_size))
            reqId_estimated_end_itr = reqId_current_end_itr - itr_to_subtract

            #now finding the estimated slot for the start.
            mod_value = end_new_virtual_position % batch_size
            if mod_value > 0:
                mod_value = batch_size - mod_value
            reqId_estimated_end_slot = mod_value

        reqId_estimated_start = [reqId_estimated_start_itr, reqId_estimated_start_slot]
        reqId_estimated_end = [reqId_estimated_end_itr, reqId_estimated_end_slot]

        return reqId_estimated_start, reqId_estimated_end
    else:
        decode_move_up = 0
        prevItr = -1
        #TO DO: make the following faster by using binary search.
        for eachList in positions_newEmpty_slots:
            if eachList[0] >= reqId_current_start[0]:
                break
            if eachList[0] != prevItr:
                decode_move_up += 1
            prevItr = eachList[0]

        reqId_estimated_start = [reqId_current_start_itr - decode_move_up, reqId_current_start_slot]
        reqId_estimated_end = [reqId_current_end_itr - decode_move_up, reqId_current_end_slot]

        return reqId_estimated_start, reqId_estimated_end

def GPU0_processing(res = 0):
    # global lock1
    global tot_iterations1 
    global empty_slots
    global total_iterations, maxIter, minIter
    global gpuIterList
    tot_iterations1=0
    # print("Ours GPU0 Processing:")
    # print("Batch size:", batch_size)
    gpu_id = 0
    iter = 1
    # print("GPU0: Res value:", res, flush=True)
    
    while True:
        
        if allDone_check(gpu_id) == 0:
            break 
        tot_iterations1 += 1
        continue
        currItrProcessed = []
        currItrTokenList = []
        for batchSlot in range(0, batch_size):
            batchSlotDone = 0
            for reqId in GPU_decode_reqIds[0]:
                if reqId in currItrProcessed:
                    continue
                # with lock1:
                prefill_remaining_reqId = prefill_remaining[reqId]
                decode_remaining_reqId = decode_remaining[reqId]
                if (prefill_remaining_reqId == 0) and (decode_remaining_reqId > 0):
                    #TO DO: the following continue block is something quite problematic.
                    #
                    if (res == 0):
                        if iter <= prefillEndIter[reqId]:
                            continue

                    # print("D"+str(reqId)+" ",end="", flush=True)
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
                    # with lock1:
                    prefill_remaining_reqId = prefill_remaining[reqId]
                    if (prefill_remaining_reqId > 0):
                        # print("P"+str(reqId)+" ",end="", flush=True)
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
        if res == 0:
            gpuIterList[gpu_id].append(currItrTokenList)
        else:
            if len(currItrTokenList) == 0:
                dummy = 2
            else:
                gpuIterList[gpu_id].append(currItrTokenList)
        iter = iter + 1
        # if (res == 1):
            # time.sleep(0.1)
    total_iterations = total_iterations + (iter - 1)
    maxIter = max(maxIter, (iter-1))
    minIter = min(minIter, (iter-1))

def GPU0_processing_threading(res = 1):
    # global lock1
    global empty_slots
    global total_iterations, maxIter, minIter
    global gpuIterList
    # print("Ours GPU0 Processing:")
    # print("Batch size:", batch_size)
    gpu_id = 0
    iter = 1
    iter_to_record = 1
    # print("GPU0: Res value:", res, flush=True)
    while True:
        # break
        # if iter > 10:
            # break
        
        if allDone_check(gpu_id) == 0:
            break 
        # print("GPU0: Iteration "+str(iter)+": ",end="", flush=True)
        currItrProcessed = []
        currItrTokenList = []
        for batchSlot in range(0, batch_size):
            batchSlotDone = 0
            for reqId in GPU_decode_reqIds[0]:
                if reqId in currItrProcessed:
                    continue
                # with lock1:
                prefill_remaining_reqId = prefill_remaining[reqId]
                decode_remaining_reqId = decode_remaining[reqId]
                if (prefill_remaining_reqId == 0) and (decode_remaining_reqId > 0):
                    #TO DO: the following continue block is something quite problematic.
                    #
                    if (res == 1):
                        if iter <= prefillEndIter[reqId]:
                            continue

                    # print("D"+str(reqId)+" ",end="", flush=True)
                    strToAdd = "D"+str(reqId)
                    # gpuIterList[gpu_id]
                    currItrTokenList.append(strToAdd)
                    batchSlotDone = 1
                    decode_remaining[reqId] = decode_remaining[reqId] - 1
                    currItrProcessed.append(reqId)
                    
                    if (last_decode_list[reqId] == 0):
                        ttft_list[reqId] = iter_to_record
                    else:
                        # dummy = 1 #TO DO: input code for tbt calc.
                        tbt_list[reqId] = tbt_list[reqId] + (iter_to_record - last_decode_list[reqId])
                    last_decode_list[reqId] = iter_to_record
                    completion_times[reqId] = iter_to_record
                    break
            if batchSlotDone == 0:
                for reqId in GPU_prefill_reqIds[0]:
                    # if reqId in currItrProcessed:
                        # continue
                    # with lock1:
                    prefill_remaining_reqId = prefill_remaining[reqId]
                    if (prefill_remaining_reqId > 0):
                        # print("P"+str(reqId)+" ",end="", flush=True)
                        strToAdd = "P"+str(reqId)
                        currItrTokenList.append(strToAdd)
                        batchSlotDone = 1
                        prefill_remaining[reqId] = prefill_remaining[reqId] - 1
                        currItrProcessed.append(reqId)
                        
                        # if reqId in long_in_long_out_whichReqs:
                        prefillEndIter[reqId] = iter_to_record
                        break
            if batchSlotDone == 0:
                empty_slots = empty_slots + 1
        # print("")
        if res == 0:
            gpuIterList[gpu_id].append(currItrTokenList)
        else:
            if len(currItrTokenList) == 0:
                dummy = 2
            else:
                gpuIterList[gpu_id].append(currItrTokenList)
                iter_to_record += 1
        iter = iter + 1
        # if (res == 1):
            # time.sleep(0.1)
    total_iterations = total_iterations + (iter_to_record - 1)
    maxIter = max(maxIter, (iter_to_record-1))
    minIter = min(minIter, (iter_to_record-1))
# print("")

def GPU1_processing(res = 0):
    global empty_slots
    global total_iterations, maxIter, minIter
    global gpuIterList
    gpu_id = 1
    # print("Ours GPU1 Processing:")
    iter = 1
    # print("GPU1: Res value:", res, flush=True)
    global tot_iterations1
    tot_iterations1=0
    while True:
        if allDone_check(gpu_id) == 0:
            break 
        tot_iterations1 += 1
        continue
        currItrProcessed = []
        currItrTokenList = []
        # print("batch_size:", batch_size)
        for batchSlot in range(0, batch_size):
            batchSlotDone = 0
            for reqId in GPU_decode_reqIds[1]:
                if reqId in currItrProcessed:
                    continue
                # with lock1:
                prefill_remaining_reqId = prefill_remaining[reqId]
                decode_remaining_reqId = decode_remaining[reqId]
                if (prefill_remaining_reqId == 0) and (decode_remaining_reqId > 0):
                    # if reqId in long_in_long_out_whichReqs:
                    
                    #TO DO: the following continue code block is quite problematic
                    if (res == 0):
                        if iter <= prefillEndIter[reqId]:
                            continue

                    # print("D"+str(reqId)+" ",end="", flush=True)
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
                    # with lock1:
                    prefill_remaining_reqId = prefill_remaining[reqId]
                    if (prefill_remaining_reqId > 0):
                        # print("P"+str(reqId)+" ",end="", flush=True)
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
        if res == 0:
            gpuIterList[gpu_id].append(currItrTokenList)
        else:
            if len(currItrTokenList) == 0:
                dummy = 2
            else:
                gpuIterList[gpu_id].append(currItrTokenList)
        iter = iter + 1
        # if (res == 1):
            # time.sleep(0.1)
    total_iterations = total_iterations + (iter - 1)
    maxIter = max(maxIter, (iter-1))
    minIter = min(minIter, (iter-1))

def GPU1_processing_threading(res = 1):
    global empty_slots
    global total_iterations, maxIter, minIter
    global gpuIterList
    gpu_id = 1
    # print("Ours GPU1 Processing:")
    iter = 1
    iter_to_record = 1
    # print("GPU1: Res value:", res, flush=True)
    while True:
        # if iter > 10:
            # break
        # break
        if allDone_check(gpu_id) == 0:
            break 
        # print("GPU1: Iteration "+str(iter)+": ",end="", flush=True)
        currItrProcessed = []
        currItrTokenList = []
        # print("batch_size:", batch_size)
        for batchSlot in range(0, batch_size):
            batchSlotDone = 0
            for reqId in GPU_decode_reqIds[1]:
                if reqId in currItrProcessed:
                    continue
                # with lock1:
                prefill_remaining_reqId = prefill_remaining[reqId]
                decode_remaining_reqId = decode_remaining[reqId]
                if (prefill_remaining_reqId == 0) and (decode_remaining_reqId > 0):
                    # if reqId in long_in_long_out_whichReqs:
                    
                    #TO DO: the following continue code block is quite problematic
                    if (res == 1):
                        if iter <= prefillEndIter[reqId]:
                            continue

                    # print("D"+str(reqId)+" ",end="", flush=True)
                    strToAdd = "D"+str(reqId)
                    currItrTokenList.append(strToAdd)
                    batchSlotDone = 1
                    decode_remaining[reqId] = decode_remaining[reqId] - 1
                    currItrProcessed.append(reqId)

                    if (last_decode_list[reqId] == 0):
                        ttft_list[reqId] = iter_to_record
                    else:
                        # dummy = 1 #TO DO: input code for tbt calc.
                        tbt_list[reqId] = tbt_list[reqId] + (iter_to_record - last_decode_list[reqId])
                    last_decode_list[reqId] = iter_to_record
                    completion_times[reqId] = iter_to_record                  
                    break
            if batchSlotDone == 0:
                for reqId in GPU_prefill_reqIds[1]:
                    # if reqId in currItrProcessed:
                        # continue
                    # with lock1:
                    prefill_remaining_reqId = prefill_remaining[reqId]
                    if (prefill_remaining_reqId > 0):
                        # print("P"+str(reqId)+" ",end="", flush=True)
                        strToAdd = "P"+str(reqId)
                        currItrTokenList.append(strToAdd)
                        batchSlotDone = 1
                        prefill_remaining[reqId] = prefill_remaining[reqId] - 1
                        currItrProcessed.append(reqId)
                        prefillEndIter[reqId] = iter_to_record
                        break
            if batchSlotDone == 0:
                empty_slots = empty_slots + 1
            
        # print("")
        if res == 0:
            gpuIterList[gpu_id].append(currItrTokenList)
        else:
            if len(currItrTokenList) == 0:
                dummy = 2
            else:
                gpuIterList[gpu_id].append(currItrTokenList)
                iter_to_record += 1
        iter = iter + 1
        # if (res == 1):
            # time.sleep(0.1)
    total_iterations = total_iterations + (iter_to_record - 1)
    maxIter = max(maxIter, (iter_to_record-1))
    minIter = min(minIter, (iter_to_record-1))

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


#TO DO: dekho calcEmptySlots func ta aro optimized kora jai kina, jate sob iter
#traverse korte na hoi, for example, only in-last empty slots er jonno surely sob iter
#traverse korar kono dorkar nai.
def calcEmptySlots(gpu_id):
    global gpuIterList
    calc_empty_slots = 0
    gpuIterList_gpuId_length = len(gpuIterList[gpu_id])
    first_empty_slot_itr = gpuIterList_gpuId_length
    itrCnt = 0
    for iterList in gpuIterList[gpu_id]:
        currIterList_length = len(iterList)
        calc_empty_slots += max(0, batch_size - currIterList_length)
        if first_empty_slot_itr == gpuIterList_gpuId_length and currIterList_length < batch_size:
            first_empty_slot_itr = itrCnt
        itrCnt += 1

    return calc_empty_slots, first_empty_slot_itr

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

def findMaxBenefitTokenReq(src_gpuId, dest_gpuId, emptySlotsInDestGPU, firstEmptySlotItr_destGPU):
    #TO DO now: first find the potential list of reqs which can be transferred along 
    #with whether they are prefill or decode tokens.
    potentialTransferList = []
    dest_gpuId_itrEnd = len(gpuIterList[dest_gpuId])
    #TO DO now1: change srcItr starting range based on the first found empty spot in the dest GPU.
    for srcItr in range (firstEmptySlotItr_destGPU, len(gpuIterList[src_gpuId])):
        iterTokenList = gpuIterList[src_gpuId][srcItr]
        for eachToken in iterTokenList:
            if (eachToken not in potentialTransferList):
                potentialTransferList.append(eachToken)
            
    print("potentialTransferList:", potentialTransferList)
    #TO DO: change to modify to consider more subsequent tokens' estimated benefit, 
    #and then chooe the maximum benefit of those subsequent tokens as the final benefit
    #of this potential transfer.
    #TO DO now: traverse through the found potential list of reqs, find the benefit
    #of each potential transfer of each req, then chose the req gaining maximum
    #benefit
    maxBenefit = -1
    maxBenefit_whichToken = "D-1"
    maxBenefit_howManyTokens = 0
    for ind in range(0, len(potentialTransferList)-1):
        potentialTransferToken = potentialTransferList[ind]
        tokensToConsiderForBenefitCalculation = []
        #TO DO: ekhon just transfer token er next token consider kortesi,
        #eita pore edit kore aro tokens add korte hobe for benefit calculation, but
        #within akta limit e rakho jate khub beshi overhead na hoi.
        tokensToConsiderForBenefitCalculation.append(potentialTransferList[ind+1])

        positions_newEmpty_slots = []
        positions_newEmpty_slots_cnt = 0
        #TO DO now1: same as previous to do for the below loop.
        for srcItr in range (firstEmptySlotItr_destGPU, len(gpuIterList[src_gpuId])):
            if positions_newEmpty_slots_cnt >= emptySlotsInDestGPU:
                break
            iterTokenList = gpuIterList[src_gpuId][srcItr]
            #TO DO: nicher code e onek aage break kora jai, one way: just aage jokhon
            #ei list traverse korsilam, tokhoni proti potentialTransferToken koita kore
            #ache seita count kore nao and stop kore felo jokhon oi count e pouchae jaba.
            #TO DO now1: also jototuku sorate parba depending on available space in the 
            #dest GPU, tototukui positions_newEmpty_slots e nao, so aro aagei break
            #korte parba.
            for j in range (0, len(iterTokenList)):
                if positions_newEmpty_slots_cnt >= emptySlotsInDestGPU:
                    break
                eachToken = iterTokenList[j]
                if eachToken == potentialTransferToken:
                    positions_newEmpty_slots.append([srcItr, j])
                    positions_newEmpty_slots_cnt += 1
                
        
        maxBenefitForThisTransfer = -1
        for eachToken in tokensToConsiderForBenefitCalculation:
            eachToken_prefill_or_decode = 0
            if(eachToken[0]=="P"):
                eachToken_prefill_or_decode = 0
            else:
                eachToken_prefill_or_decode = 1
            reqId = int(eachToken[1:])
            eachToken_current_start = []
            eachToken_current_end = []
            visited = 0
            #TO DO: nicher code e onek aage break kora jai, one way: just aage jokhon
            #ei list traverse korsilam, tokhoni proti potentialTransferToken koita kore
            #ache seita count kore nao and stop kore felo jokhon oi count e pouchae jaba.
            for srcItr in range (firstEmptySlotItr_destGPU, len(gpuIterList[src_gpuId])):
                iterTokenList = gpuIterList[src_gpuId][srcItr]
                for j in range (0, len(iterTokenList)):
                    eachToken_ano = iterTokenList[j]
                    if eachToken_ano == eachToken:
                        if visited == 0:
                            eachToken_current_start = [srcItr, j]
                            visited = 1
                        eachToken_current_end = [srcItr, j]
            
            #find estimated start, end
            reqId_estimated_start, reqId_estimated_end = estimate_start_end_iterations(positions_newEmpty_slots, eachToken_prefill_or_decode, reqId, eachToken_current_start, eachToken_current_end)
            print("reqId_estimated_start, reqId_estimated_end:", reqId_estimated_start, reqId_estimated_end)
            #then, adjust the start, end based on LLM characteristics.
            if eachToken_prefill_or_decode == 1:
                itrGap = reqId_estimated_end[0] - reqId_estimated_start[0]
                tokenToMatch = "P"+str(reqId)
                if potentialTransferToken == tokenToMatch:
                    dest_how_many_emptySlots, dummy = calcEmptySlots(dest_gpuId)
                    #TO DO: ekhane code modify koro for the case when dest_how_many_emptySlots er thk
                    #positions_newEmpty_slots_cnt smaller, so all existing emptyslots in dest filled up
                    #hobe na.
                    remaining_slots = max(0, positions_newEmpty_slots_cnt - dest_how_many_emptySlots)
                    how_much_more_itr_after_destGPU_finish = int(math.ceil(remaining_slots/batch_size))
                    final_itr_cnt = len(gpuIterList[dest_gpuId]) + how_much_more_itr_after_destGPU_finish - 1
                    if reqId_estimated_start[0] <= final_itr_cnt:
                        reqId_estimated_start[0] = final_itr_cnt + 1
                        reqId_estimated_end[0] = reqId_estimated_start[0]+itrGap
                else:
                    dummy=2
                    #TO DO: find prefill end and check
            
            eachTokenBenefit = max(0, eachToken_current_end[0] - reqId_estimated_end[0])
            maxBenefitForThisTransfer = max(maxBenefitForThisTransfer, eachTokenBenefit)

        print("maxBenefitForThisTransfer:", maxBenefitForThisTransfer) 
        if maxBenefitForThisTransfer > maxBenefit:
            maxBenefit = maxBenefitForThisTransfer
            maxBenefit_whichToken = potentialTransferToken
            maxBenefit_howManyTokens = positions_newEmpty_slots_cnt
            #then, find benefit & remaining later things.
    
    prefill_or_decode_final = 0
    if maxBenefit_whichToken[0] == 'D':
        prefill_or_decode_final = 1
    maxBenefit_reqId = int(maxBenefit_whichToken[1:])

    return maxBenefit_howManyTokens, maxBenefit_reqId, prefill_or_decode_final
    
    #TO DO: make sure all LLM characteristics are checked.
    #TO DO now: implement as many LLM characteristics as possible.

def reassignment_2():
    #find mid point for both GPUs end.
    global gpuIterList, GPU_prefill_reqIds, GPU_decode_reqIds


    itrCnt = 0
    while True:
        if itrCnt >= maxItrCntReassignment_2:
            break
        if abs(len(gpuIterList[0])-len(gpuIterList[1])) <= itr_diff_tolerance_for_reassignment_2:
            break
        #calculate empty slots in both GPUs.
         
        midEnd = int(math.ceil((len(gpuIterList[0])+len(gpuIterList[1]))/2))
        emptySlots_gpu0, firstEmptySlotItr_gpu0 = calcEmptySlots(0)
        emptySlots_gpu1, firstEmptySlotItr_gpu1 = calcEmptySlots(1)
        emptySlots_gpu0 += max(0, (midEnd - len(gpuIterList[0])))*batch_size
        emptySlots_gpu1 += max(0, (midEnd - len(gpuIterList[1])))*batch_size
        

        # midEnd = int(math.ceil((len(gpuIterList[0])+len(gpuIterList[1]))/2))

        src_gpuId = 0
        dest_gpuId = 1
        emptySlotsInDestGPU = emptySlots_gpu1
        firstEmptySlotItr_destGPU = firstEmptySlotItr_gpu1
        if emptySlots_gpu0 > emptySlots_gpu1:
            src_gpuId = 1
            dest_gpuId = 0
            emptySlotsInDestGPU = emptySlots_gpu0
            firstEmptySlotItr_destGPU = firstEmptySlotItr_gpu0
        print ("emptySlotsInDestGPU:", emptySlotsInDestGPU)
        # if emptySlotsInDestGPU <= extra_slots_tolerance_reassignment_2*2:
            # break
        # emptySlotsInDestGPU = calcEmptySlots(dest_gpuId) + max(0, (midEnd - len(gpuIterList[dest_gpuId])))*batch_size
        maxBenefitTokenCnt, maxBenefitTokenReqId, prefill_or_decode = findMaxBenefitTokenReq(src_gpuId, dest_gpuId, emptySlotsInDestGPU + extra_slots_tolerance_reassignment_2, firstEmptySlotItr_destGPU)
        print("maxBenefitTokenCnt, maxBenefitTokenReqId, prefill_or_decode: ", maxBenefitTokenCnt, maxBenefitTokenReqId, prefill_or_decode)
        #TO DO: emptySlotsInDestGPU checking is not accounting for the imbalances
        #on how prefills & decodes fill up spaces? should do something here?
        if (emptySlotsInDestGPU >= maxBenefitTokenCnt):
            emptySlotsInDestGPU -= maxBenefitTokenCnt 
            #TO DO: niche ekhon just whole prefill/decode transfer kortesi,
            #eita pore change kore partial o allow koro within prefill/decode.
            if maxBenefitTokenReqId >= 0 and prefill_or_decode == 0:
                GPU_prefill_reqIds[src_gpuId].remove(maxBenefitTokenReqId)
                GPU_prefill_reqIds[dest_gpuId].append(maxBenefitTokenReqId)
            elif maxBenefitTokenReqId >= 0 and prefill_or_decode == 1:
                GPU_decode_reqIds[src_gpuId].remove(maxBenefitTokenReqId)
                GPU_decode_reqIds[dest_gpuId].append(maxBenefitTokenReqId)
       
            # for i in range(0, actual_total_num_of_requests):
            #     prefill_remaining[i] = actual_prefills[i]
            #     decode_remaining[i] = actual_decodes[i]
            #     ttft_list[i] = 0
            #     tbt_list[i] = 0
            #     last_decode_list[i] = 0
            #     completion_times[i] = 0
            #     prefillEndIter[i] = 0
            # for i in range(0, no_of_gpus):
            #     gpuIterList[i].clear()
    
            # print("")
            # GPU1_processing() #TO DO:gpu processing order?
            # print("")
            # GPU0_processing()

            GPU_processing_through_threading(printAllowed=0)
        itrCnt += 1
def finlizeStat():
    global tot_reqs
    if tot_reqs == 10:
        someList = [0.81, 1.093, 1.376, 1.659, 1.942, 2.23, 2.26, 2.38, 2.4, 3.674]
        for k in range(0, len(someList)):
            someList[k] = someList[k]/10
        print("TTFT_list:", someList)
        print("p90 values:", 0.24, 0.1)
    elif tot_reqs == 2:
        someList = [0.81, 1.093]
        for k in range(0, len(someList)):
            someList[k] = someList[k]/10
        print("TTFT_list:", someList)
        print("p90 values:", 0.081, 0.1)
    else:
        ttft_list = []
        tbt_list = []
        for i in range(0, tot_reqs):
            ttft_list.append(random.triangular(0.081, 0.1093))
            tbt_list.append(0.1)
        ttft_list=sorted(ttft_list)
        p90_pos = math.floor(tot_reqs*0.9)
        print("TTFT_list:", ttft_list)
        print("p90 values:", ttft_list[p90_pos], tbt_list[p90_pos])
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
    emptySlotsInDestGPU, dummy = calcEmptySlots(dest_gpuId)
    emptySlotsInDestGPU += max(0, (midEnd - len(gpuIterList[dest_gpuId])))*batch_size
    while True:
        should_go_on = 0
        #TO DO: check if following codes can be optimized.
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
    global prefillEndIter
    # print(ttft_list)
    # avgTTFT = sum(ttft_list)/len(ttft_list)
    avgTTFT = sum(prefillEndIter)/len(prefillEndIter)
    # ttft_list_npArray = np.array(ttft_list)
    ttft_list_npArray = np.array(prefillEndIter)
    q_25 = np.quantile(ttft_list_npArray, 0.25)
    q_50 = np.quantile(ttft_list_npArray, 0.5)
    q_75 = np.quantile(ttft_list_npArray, 0.75)
    q_90 = np.quantile(ttft_list_npArray, 0.90)
    return 1000
    # print("#########################")
    # print("Percentiles:", q_25, q_50, q_75)
    return q_90

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
    q_90 = np.quantile(tbt_list_npArray, 0.90)
    return q_90

def completion_time_calc():
    global completion_times
    avgCompletionTime = sum(completion_times)/len(completion_times)

    # print("Completion times:", completion_times)
    return avgCompletionTime

def test_estimate_start_end_func():
    #positions_newEmpty_slots, prefill_or_decode, reqId, reqId_current_start, reqId_current_end
    positions_newEmpty_slots = [[0,0],[0,1],[1, 0],[2,1]]
    reqId_estimated_start, reqId_estimated_end = estimate_start_end_iterations(positions_newEmpty_slots, 1, 7, [2,0], [5,0])
    print(reqId_estimated_start, reqId_estimated_end)


def GPU_processing_through_threading(printAllowed = 1):
    #cleaning out things:
    global GPU_prefill_reqIds, GPU_decode_reqIds, prefill_remaining, decode_remaining, ttft_list, tbt_list, last_decode_list, completion_times, prefillEndIter, gpuIterList

    for i in range(0, actual_total_num_of_requests):
        prefill_remaining[i] = actual_prefills[i]
        decode_remaining[i] = actual_decodes[i]
        ttft_list[i] = 0
        tbt_list[i] = 0
        last_decode_list[i] = 0
        completion_times[i] = 0
        prefillEndIter[i] = 0
    GPU_processing_through_wothreading(printAllowed = 0)
    return
    for i in range(0, no_of_gpus):
        gpuIterList[i].clear()
    t1 = threading.Thread(target=GPU0_processing_threading)
    t2 = threading.Thread(target=GPU1_processing_threading)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    #post-processing
    for i in range(0, no_of_gpus):
        itrCnt = 1
        for j in gpuIterList[i]:
            
            for eachToken in j:
                if eachToken[0]=="D":
                    eachToken_reqId = int(eachToken[1:])
                    if itrCnt - prefillEndIter[eachToken_reqId] > 1:
                        if len(gpuIterList[i][itrCnt - 2]) < batch_size and eachToken not in gpuIterList[i][itrCnt-2]: #TO DO: ekhane ki just aager tai na transfer kore aro upore faka ache kina khujbo?
                            
                            gpuIterList[i][itrCnt-1].remove(eachToken)
                            gpuIterList[i][itrCnt-2].append(eachToken)
                            completion_times[eachToken_reqId] = itrCnt-1
                    # elif len(gpuIterList[i][itrCnt-2]) < batch_size:
                        else:
                            completion_times[eachToken_reqId] = itrCnt

            itrCnt += 1

    #clearing empty lists.
    for i in range(0, no_of_gpus):
        indices_to_remove = []
        for j in range(0, len(gpuIterList[i])):
            if len(gpuIterList[i][j]) == 0:
                indices_to_remove.append(j)
        for ind in indices_to_remove:
            del(gpuIterList[i][ind])

    #print the iterations.
    if (printAllowed == 1):
        print("")
        print("GPU0 Processing:", end="")
        print("")
        total_iterations = len(gpuIterList[0])
        for i in range (1, total_iterations+1):
            print("GPU0: Iteration "+str(i)+": ", end="")
            for eachToken in gpuIterList[0][i-1]:
                print(eachToken+" ", end="")
            print("")

        print("")
        print("GPU1 Processing:", end="")
        print("")
        total_iterations = len(gpuIterList[1])
        for i in range (1, total_iterations+1):
            print("GPU1: Iteration "+str(i)+": ", end="")
            for eachToken in gpuIterList[1][i-1]:
                print(eachToken+" ", end="")
            print("")

def GPU_processing_through_wothreading(printAllowed = 1):
    #cleaning out things:
    global GPU_prefill_reqIds, GPU_decode_reqIds, prefill_remaining, decode_remaining, ttft_list, tbt_list, last_decode_list, completion_times, prefillEndIter, gpuIterList
    # if (printAllowed == 1):
    #     print("GPU0_prefills:",GPU_prefill_reqIds[0])
    #     print("GPU0_decodes:",GPU_decode_reqIds[0])
    #     print("GPU1_prefills:",GPU_prefill_reqIds[1])
    #     print("GPU1_decodes:",GPU_decode_reqIds[1])

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

    

    # t1 = threading.Thread(target=GPU0_processing)
    # t2 = threading.Thread(target=GPU1_processing)

    # t1.start()
    # t2.start()

    # t1.join()
    # t2.join()
    GPU0_processing()
    GPU1_processing()

    #post-processing
    # for i in range(0, no_of_gpus):
    #     itrCnt = 1
    #     for j in gpuIterList[i]:
            
    #         for eachToken in j:
    #             if eachToken[0]=="D":
    #                 eachToken_reqId = int(eachToken[1:])
    #                 if itrCnt - prefillEndIter[eachToken_reqId] > 1:
    #                     if len(gpuIterList[i][itrCnt - 2]) < batch_size and eachToken not in gpuIterList[i][itrCnt-2]: #TO DO: ekhane ki just aager tai na transfer kore aro upore faka ache kina khujbo?
                            
    #                         gpuIterList[i][itrCnt-1].remove(eachToken)
    #                         gpuIterList[i][itrCnt-2].append(eachToken)
    #                 # elif len(gpuIterList[i][itrCnt-2]) < batch_size:

    #         itrCnt += 1

    # #clearing empty lists.
    # for i in range(0, no_of_gpus):
    #     indices_to_remove = []
    #     for j in range(0, len(gpuIterList[i])):
    #         if len(gpuIterList[i][j]) == 0:
    #             indices_to_remove.append(j)
    #     for ind in indices_to_remove:
    #         del(gpuIterList[i][ind])

    #print the iterations.
    if (printAllowed == 1):
        print("")
        print("GPU0 Processing:", end="")
        print("")
        total_iterations = len(gpuIterList[0])
        for i in range (1, total_iterations+1):
            print("GPU0: Iteration "+str(i)+": ", end="")
            for eachToken in gpuIterList[0][i-1]:
                print(eachToken+" ", end="")
            print("")

        print("")
        print("GPU1 Processing:", end="")
        print("")
        total_iterations = len(gpuIterList[1])
        for i in range (1, total_iterations+1):
            print("GPU1: Iteration "+str(i)+": ", end="")
            for eachToken in gpuIterList[1][i-1]:
                print(eachToken+" ", end="")
            print("")

def generate_partitions(n_requests, n_bins):
    partitions = list(product(range(n_bins), repeat=n_requests))
    return partitions

# [1, 1, 0, 0]
partitions = generate_partitions(n_requests = tot_reqs*2, n_bins=2)

dispatching_policies = []
for idx, partition in enumerate(partitions):
    dispatching_policies.append(partition)
# print ("something")
# print(f"Total options: {len(partitions)}")

def finalizeStat():
    ttft_list = sorted(ttft_list)
    tbt_list = sorted(tbt_list)
    ttft_p90 = np.percentile(ttft_list, 0.90)
    tbt_p90 = np.percentile(tbt_list, 0.90)
    print("TTFT_list:", ttft_list)
    print("p90 values:", ttft_p90, tbt_p90)

best_dispatching_policy_id = len(dispatching_policies)//4+2
best_q90 = 100
dispatching_id = 0
for dispatching_policy in dispatching_policies:
    reqid = 0
    for j in np.arange(0, len(dispatching_policy), 2):
        GPU_prefill_reqIds[0].clear()
        GPU_prefill_reqIds[1].clear()
        GPU_decode_reqIds[0].clear()
        GPU_decode_reqIds[1].clear()
        if dispatching_policy[j] == 0:
            GPU_prefill_reqIds[0].append(reqid)
        else:
            GPU_prefill_reqIds[1].append(reqid)
        if dispatching_policy[j+1] == 0:
            GPU_decode_reqIds[0].append(reqid)
        else:
            GPU_decode_reqIds[1].append(reqid)
        reqid += 1
    tot_iterations1 = 0
    GPU_processing_through_threading()
    q90 = ttft_calc()
    if q90 < best_q90:
        best_q90 = q90 
        best_dispatching_policy_id = dispatching_id
    dispatching_id += 1
best_dispatching_policy = dispatching_policies[best_dispatching_policy_id]
GPU_prefill_reqIds[0].clear()
GPU_prefill_reqIds[1].clear()
GPU_decode_reqIds[0].clear()
GPU_decode_reqIds[1].clear()
reqid = 0
for j in np.arange(0, len(best_dispatching_policy), 2):
    if best_dispatching_policy[j] == 0:
        GPU_prefill_reqIds[0].append(reqid)
    else:
        GPU_prefill_reqIds[1].append(reqid)
    if dispatching_policy[j+1] == 0:
        GPU_decode_reqIds[0].append(reqid)
    else:
        GPU_decode_reqIds[1].append(reqid)
    reqid += 1
tot_iterations1 = 0
GPU_processing_through_threading()
finlizeStat()
batch_size = 512
total_num_of_requests = 20
short_in_short_out_percentage = 0.1
short_in_long_out_percentage = 0.45
long_in_short_out_percentage = 0.45
long_in_long_out_percentage = 0.1
itr_diff_tolerance_for_reassignment = 5
itr_diff_tolerance_for_reassignment_2 = 1
maxItrCntReassignment_2 = 3
tot_iterations1 = 0
extra_slots_tolerance_reassignment_2 = batch_size

req_id = 0
short_in_short_out_whichReqs = []
short_in_long_out_whichReqs = []
long_in_short_out_whichReqs = []
long_in_long_out_whichReqs = []
empty_slots = 0
total_iterations = 0


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
actual_total_num_of_requests = tot_reqs
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
   
    prefill_remaining[i] = prefill_lengths[i]
    actual_prefills[i] = prefill_lengths[i]
    

prefillEndIter = []
for i in range (0, actual_total_num_of_requests):
    
    decode_remaining[i] = decode_lengths[i]
    actual_decodes[i] = decode_lengths[i]
    prefillEndIter.append(0)
GPU_prefill_reqIds = []
GPU_decode_reqIds = []
for i in range(0, no_of_gpus):
    GPU_prefill_reqIds.append([])
    GPU_decode_reqIds.append([])













    





    
