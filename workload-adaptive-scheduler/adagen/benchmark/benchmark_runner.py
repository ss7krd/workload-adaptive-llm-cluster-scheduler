import sys
sys.path.append("/home/shubhasu/shubhasu/sarathi-serve")
sys.path.append("/usr/local/lib/python3.10/dist-packages")
sys.path.append("/usr/lib/python3/dist-packages")

import logging
import os
import time

import ray
import ray.util.collective as col

import torch
import cupy

import wandb
from tqdm import tqdm

from sarathi import LLMEngine, SamplingParams
from sarathi.benchmark.config import BenchmarkConfig
from sarathi.benchmark.entities import Request
from sarathi.benchmark.request_generator import RequestGeneratorRegistry
from sarathi.benchmark.utils.random import set_seeds
from sarathi.config import ReplicaConfig
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.types import ReplicaResourceMapping, ResourceMapping
from sarathi.utils import get_ip

from sarathi.benchmark.global_variable_tester import MyClass

logger = logging.getLogger(__name__)


class BenchmarkRunner:

    class_variable = 10


    
    def __init__(
        self,
        replica_id: int,
        config: BenchmarkConfig,
        resource_mapping: ResourceMapping,
    ) -> None:
        self.replica_id = replica_id
        self.config = config
        self.resource_mapping = resource_mapping
        self.requests = []

        replica_config = ReplicaConfig(
            replica_id,
            self.config.output_dir,
            resource_mapping,
        )
        os.makedirs(replica_config.output_dir, exist_ok=True)

        # set_seeds(self.config.seed)
        # request_generator = RequestGeneratorRegistry.get(
        #     self.config.request_generator_config.get_type(),
        #     self.config.request_generator_config,
        # )
        
        #THIS LINE WAS IMPORTANT
        # self.requests = requests

        # select every nth request for this replica
        # e.g. if there are 4 replicas, and this is the 2nd replica, then
        # we will select the 2nd, 6th, 10th, ... requests
        # round robin scheduling
        # self.requests = self.requests[self.replica_id :: self.config.num_replicas]

        if self.config.num_replicas > 1:
            # disable per-replica wandb logging for multi-replica runs
            # so that we can aggregate metrics across all replicas
            self.config.metrics_config.wandb_project = None

        system_config = self.config.create_system_config(replica_config)
        self.llm_engine = LLMEngine.from_system_config(system_config)

        if wandb.run is not None:
            wandb.config.update(self.config.to_dict())

        #TO DO Sudipta: later remove/modify:
        # col.init_collective_group(2, self.replica_id, "nccl")
        # print("ray group is initialized")

    def set_requests(self, requests):
        self.requests = requests

    def append_a_request(self, request):
        self.requests.append(request)

    def set_seq_id_list(self, seq_id_list):
        self.seq_id_list = seq_id_list

    def set_seq_category_list(self, seq_category_list):
        self.seq_category_list = seq_category_list

    def set_prefill_how_many_in_this_replica_list(self, prefill_how_many_in_this_replica_list):
        self.prefill_how_many_in_this_replica_list = prefill_how_many_in_this_replica_list

    def set_decode_how_many_in_this_replica_list(self, decode_how_many_in_this_replica_list):
        self.decode_how_many_in_this_replica_list = decode_how_many_in_this_replica_list

    def set_current_replica_id_list(self, current_replica_id_list):
        self.current_replica_id_list = current_replica_id_list

    def set_prev_seq_id_list(self, prev_seq_id_list):
        self.prev_seq_id_list = prev_seq_id_list

    def set_prev_seq_replica_id_list(self, prev_seq_replica_id_list)
        self.prev_seq_replica_id_list = prev_seq_replica_id_list


    @classmethod
    def change_class_variable(cls, new_value=0):
        cls.class_variable = new_value
    
    @classmethod
    def access_class_variable(cls):
        return cls.class_variable

    def _get_input_params(
        self, request: Request, first_request_time: float, seq_id: int, seq_category: int, prefill_how_many_in_this_replica:int, decode_how_many_in_this_replica:int, current_replica_id:int, prev_seq_id:int, prev_seq_replica_id:int
    ) -> SamplingParams:
        sampling_params = SamplingParams(
            ignore_eos=True,
            max_tokens=request.num_decode_tokens,#TO DO Sudipta: eita ki performance degrading hote pare?
            temperature=0,
            top_p=1.0,
        )
        prompt_token_ids = [1] * request.num_prefill_tokens
        prompt = None
        if seq_category == 13:
            prompt_token_ids = None
            prompt = "You"
        
        return {
            "seq_id": seq_id,
            "prompt": prompt,
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": sampling_params,
            "arrival_time": first_request_time + request.arrived_at,
            "seq_category": seq_category,
            "prefill_how_many_in_this_replica": prefill_how_many_in_this_replica,
            "decode_how_many_in_this_replica": decode_how_many_in_this_replica,
            "current_replica_id": current_replica_id,
            "prev_seq_id": prev_seq_id,
            "prev_seq_replica_id": prev_seq_replica_id

        }

    def warmup(self) -> None:
        self.llm_engine.add_request(**self._get_input_params(self.requests[0], 0, 0, 0,-2, -2, 0, -1, -1))

        is_completed = False
        while not is_completed:
            step_outputs = self.llm_engine.step()
            is_completed = step_outputs[0].finished

        self.llm_engine.reset_metrics()

    def _run(self) -> None:
        if self.config.enable_profiling:
            self.llm_engine.start_profiling()

        num_processed_requests = 0
        num_steps = 0
        pbar = tqdm(
            total=len(self.requests),
            desc=f"Replica {self.replica_id} processed requests",
        )
        start_time = time.monotonic()

        # Run the engine.
        while num_processed_requests < len(self.requests):

            
            elapsed_time = time.monotonic() - start_time
            if elapsed_time > self.config.time_limit:
                break

            step_outputs = self.llm_engine.step()
            num_steps += 1

            for output in step_outputs:
                if output.finished:
                    num_processed_requests += 1
                    pbar.update(1)
            
            # if self.replica_id == 1:
            #     # BenchmarkRunner.class_variable += 1
            #     # print(f"Replica {self.replica_id} is accessing: {BenchmarkRunner.access_class_variable()}")
            #     # continue
            #     # BenchmarkRunner.change_class_variable(num_processed_requests)

            #     with open("hello.txt",'w') as myFile:
            #         myFile.write(f"something by replica 1: {num_processed_requests}")

            # # else:
            # else:
            #     # print(f"Replica {self.replica_id} is reading: {BenchmarkRunner.access_class_variable()}")
            #     if os.path.exists("hello.txt"):
            #         with open("hello.txt",'r') as myFile:
            #             for eachLine in myFile:
            #                 print(f"Replica 0 is reading: {eachLine}")


                

        end_time = time.monotonic()
        pbar.close()

        logger.info(
            f"Replica {self.replica_id} exiting after processing {len(self.requests)} ({num_steps} iterations), Total time taken: {end_time - start_time:.2f} seconds"
        )

        if self.config.enable_profiling:
            self.llm_engine.stop_profiling()

    def _add_requests(self) -> None:
        index = 0
        first_request_time = time.monotonic()
        while index < len(self.requests):
            request = self.requests[index]
            # seq_category = 
            self.llm_engine.add_request(
                **self._get_input_params(request, first_request_time, self.seq_id_list[index], self.seq_category_list[index], self.prefill_how_many_in_this_replica_list[index], self.decode_how_many_in_this_replica_list[index], self.current_replica_id_list[index], self.prev_seq_id_list[index], self.prev_seq_replica_id_list[index])
            )
            index += 1

    def run(self) -> None:
        self.llm_engine.reset_metrics()
        self._add_requests()
        self._run()
        self.llm_engine.pull_worker_metrics()
        metric_store = self.llm_engine.get_metric_store()
        return metric_store

    def do_send(self, target_rank): #TO DO Sudipta: add group_name, do something for blocking calls like creating separate streams.
        # tensor_to_send = torch.tensor(list_to_send, device='cuda:1')
        print("reached send function.")
        local_rank = self.resource_mapping[0][1]
        tensor_to_send = torch.rand(4, device=torch.device(f"cuda:{local_rank}"))
        # print(tensor_to_send.cpu())
        print("ending sending 1")
        col.send(tensor_to_send, target_rank)
        print("ending sending 2")
        # return 1

    def do_recv(self, src_rank):
        # recv_list = [0, 0, 0, 0]
        # recv_tensor = torch.tensor(recv_list, device='cuda:0')
        print("reached recv. funtion")
        local_rank = self.resource_mapping[0][1]
        recv_tensor = torch.rand(4, device=torch.device(f"cuda:{local_rank}"))
        print("ending receiving 1")
        col.recv(recv_tensor, src_rank)
        print("ending receiving 2")#
        # print(recv_tensor.cpu())
        # return recv_tensor

class LLMInstance:
    def __init__(self, id):
        self.id = id
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)

    def __repr__(self):
        return f"Instance {self.id} (requests: {len(self.requests)})"   

  

class BenchmarkRunnerLauncher:
    #COMPLETE.-calling the following for each s_t.
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        Request_generator = RequestGeneratorRegistry.get(
            self.config.request_generator_config.get_type(),
            self.config.request_generator_config,
        )
        self.requests = request_generator.generate()
        model = torch.load_state_dict(torch.load("distillBert.pth"))
        model.eval()
        for req in self.requests:
            with torch.no_grad():
                req.decode_length = model(req)
        batch_limit_start = 0
        batch_limit_end = self.config.st_size
        while batch_limit_end <= len(requests):
            runner_eachBatch = BenchmarkRunnerLauncher_eachBatch(self.config, self.requests[batch_limit_start, batch_limit_end])
            batch_limit_start = batch_limit_end
            batch_limit_end = batch_limit_end + self.config.st_size 



class BenchmarkRunnerLauncher_eachBatch:

    def __init__(self, config: BenchmarkConfig, requests) -> None:
        self.config = config
        self.is_multi_replica = self.config.num_replicas > 1
        self.requests = requests
        ray.init(ignore_reinit_error=True)

        self._validate_cluster_resources()
        self.runners = self._create_runners()

        if self.is_multi_replica:
            # self.aggregate_metric_store = self._create_aggregate_metric_store()
            dummy=1

    def dispatch_requests(self, instances, requests, prefill_threshold=100, decode_threshold=100):

        n = len(self.config.instances)
        assert n % 2 == 0, "Number of instances must be even"

        group_a = instances[:n // 2]
        group_b = instances[n // 2:]

        for request in requests:
            prefill = request['prefill_length']
            decode = request['decode_length']

            self.config.prefill_threshold = np.percentile(self.prompt_length_list, 0.75)
            self.config.decode_threshold = np.percentile(self.response_length_list, 0.75)

            if prefill > self.config.prefill_threshold or decode < self.config.decode_threshold:
                target_group = group_a
            else:
                target_group = group_b

            
            instance = random.choice(target_group)
            instance.handle_request(request)

        return instances 

    def _validate_cluster_resources(self):
        num_replicas = self.config.num_replicas
        num_gpus_required = num_replicas * self.config.parallel_config.world_size

        available_resources = ray.available_resources()

        assert (
            available_resources["GPU"] >= num_gpus_required
        ), f"Insufficient GPUs. Required: {num_gpus_required}, Available: {available_resources['GPU']}"

    def _get_replica_resource_mapping(self) -> ReplicaResourceMapping:
        
        if self.config.replica_resource_mapping:
            assert len(self.config.replica_resource_mapping) == self.config.num_replicas
            logger.info(
                f"Replica resource mapping: {self.config.replica_resource_mapping}"
            )
            return self.config.replica_resource_mapping

        cluster_resources_keys = list(ray.available_resources().keys())
        num_gpus = ray.available_resources()["GPU"]
        ip_addresses = [
            x
            for x in cluster_resources_keys
            if x.startswith("node:") and x != "node:__internal_head__"
        ]

        runner_ip = f"node:{get_ip()}"

        ip_addresses.remove(runner_ip)
        ip_addresses.insert(0, runner_ip)

        num_nodes = len(ip_addresses)
        assert num_nodes > 0, "No nodes found in the cluster"
        assert num_gpus > 0, "No GPUs found in the cluster"
        assert (
            num_gpus % num_nodes == 0
        ), f"Number of GPUs ({num_gpus}) is not a multiple of number of nodes ({num_nodes})"
        num_gpus_per_node = int(num_gpus // num_nodes)
        num_replicas = self.config.num_replicas
        num_gpus_per_replica = self.config.parallel_config.world_size

        assert (
            num_gpus >= num_replicas * num_gpus_per_replica
        ), f"Insufficient GPUs. Required: {num_replicas * num_gpus_per_replica}, Available: {num_gpus}"

        replica_resource_mapping = []

        available_gpus = []
        for ip_address in ip_addresses:
            for gpu_id in reversed(range(num_gpus_per_node)):
                available_gpus.append((ip_address, gpu_id))

        for _ in range(num_replicas):
            resource_mapping = []
            for _ in range(num_gpus_per_replica):
                resource_mapping.append(available_gpus.pop(0))
            replica_resource_mapping.append(resource_mapping)

        logger.info(f"Replica resource mapping: {replica_resource_mapping}")

        return replica_resource_mapping

    def length_aware_scheduling(self):
        assignment_scheduling, compute_layouts_list = dispatch_requests(self.instances, self.requests)
        return assignment_scheduling, compute_layouts_list

    def enough_memory_check(self, instance):
        KV_cache_size = self.running_count_active_tokens_list[instance]*self.config.eachTokenSize
        if KV_cache_size > self.config.KV_cache_size:
            return True 
        else:
            return False

    def is_req_completed(self, instance_compute_layout, req):
        token_cnt = 0
        for iteration in range(0, len(instance_compute_layout)):
            for eachLetter in iteration:
                if eachLetter[1]==req.req_id:
                    token_cnt +=1 
        if token_cnt >= req.output_tokens_count:
            return True
        else:
            return False


    def simulate(self, intermediate_assignment, compute_layouts_list):
        # global lock1
        # compute_layouts_list = []
        # running_count_active_tokens_list = []
        # for instance in range(0, self.config.instance_count):
        #     running_count_active_tokens_list.append(0)
        for instance in range(0, self.config.instance_count):
            instance_compute_layout = compute_layouts_list[instance]
            for req in self.requests:
                if is_req_completed(instance_compute_layout, req):
                    self.running_count_active_tokens_list[instance] -= req.output_tokens_count

        for instance in range(0, self.config.instance_count):
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
                            if enough_memory_check(instance):
                                strToAdd = "D"+str(reqId)
                                self.running_count_active_tokens_list[instance] += 1
                            curr_instance_computeLayout = self.config.compute_layout_func(strToAdd, instance)
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
                                if enough_memory_check(instance):
                                    strToAdd = "P"+str(reqId)
                                    self.running_count_active_tokens_list[instance] += 1
                                curr_instance_computeLayout = self.config.compute_layout_func(strToAdd, instance)
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

            compute_layouts_list.append(curr_instance_computeLayout)

        return compute_layouts_list
    def calcEmptySlots(self, gpu_id):
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
    def distribution_aware_scheduling(self,compute_layouts_list):
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
        src_gpuId_list = []
        dest_gpuId_list = []
        score_list = []
        gpuId_list = []
        for gpuId in range(0, self.config.instance_count):
            gpuId_list.append(gpuId)
            curr_score = self.total_iteration_count(compute_layouts_list[gpuId])
            score_list.append(curr_score)
        gpuId_score_pairs = list(zip(gpuId_list, score_list))
        gpuId_score_pairs.sort(reverse=True, key=lambda x: x[1])
        src_gpuId = gpuId_score_pairs[0]
        dest_gpuId = gpuId_score_pairs[self.config.instance_count-1]
        while src_gpuId < dest_gpuId:
            src_gpuId_list.append(src_gpuId)
            dest_gpuId_list.append(dest_gpuId)
            src_gpuId += 1
            dest_gpuId -= 1
        
        dest_gpuId_index = 0
        for src_gpuId_index in range(0, len(src_gpuId_list)):
            src_gpuId = src_gpuId_list[src_gpuId_index]
            dest_gpuId = dest_gpuId_list[dest_gpuId_index]
            dest_gpuId_index += 1
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

        return intermediate_scheduling_res, updated compute_layouts_list

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

    def findMaxBenefitTokenReq(src_gpuId, dest_gpuId, clusterRepresentatives, emptySlotsInDestGPU, firstEmptySlotItr_destGPU):
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
        potentialTransferList = clusterRepresentatives
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
    def make_clusters(self, gpuId):
        req_list = self.create_req_list(gpuId)
        req_list_details = []
        for reqId in req_list:
            req_list_details.append([reqId, reqId.prefill_token_count, self.impacted_requests_token_count(reqId), self.impacted_requests_prefills_last_itr(reqId)])
            req_list_details.append([reqId, reqId.decode_token_count, self.impacted_requests_token_count(reqId), self.impacted_requests_prefills_last_itr(reqId)])
        dbscan = DBSCAN(eps=1e-7)
        clusters = dbscan.fit_predict(req_list_details)
        return clusters

    def selective_distributed(compute_layouts_list):

    
        global gpuIterList, GPU_prefill_reqIds, GPU_decode_reqIds

        src_gpuId_list = []
        dest_gpuId_list = []
        score_list = []
        gpuId_list = []
        for gpuId in range(0, self.config.instance_count):
            gpuId_list.append(gpuId)
            curr_score = self.total_prefill_count(compute_layouts_list[gpuId])+self.total_decode_count(compute_layouts_list[gpuId])+self.avg_prefills_last_itr(compute_layouts_list[gpuId])
            score_list.append(curr_score)
        gpuId_score_pairs = list(zip(gpuId_list, score_list))
        gpuId_score_pairs.sort(reverse=True, key=lambda x: x[1])
        src_gpuId = gpuId_score_pairs[0]
        dest_gpuId = gpuId_score_pairs[self.config.instance_count-1]
        while src_gpuId < dest_gpuId:
            src_gpuId_list.append(src_gpuId)
            dest_gpuId_list.append(dest_gpuId)
            src_gpuId += 1
            dest_gpuId -= 1
        
        dest_gpuId_index = 0
        for src_gpuId_index in range(0, len(src_gpuId_list)):
            src_gpuId = src_gpuId_list[src_gpuId_index]
            dest_gpuId = dest_gpuId_list[dest_gpuId_index]
            dest_gpuId_index += 1
            itrCnt = 0
            while True:
                if itrCnt >= maxItrCntReassignment_2:
                    break
                if abs(len(gpuIterList[0])-len(gpuIterList[1])) <= itr_diff_tolerance_for_reassignment_2:
                    break
                
                
                midEnd = int(math.ceil((len(gpuIterList[0])+len(gpuIterList[1]))/2))
                emptySlots_gpu0, firstEmptySlotItr_gpu0 = calcEmptySlots(src_gpuId)
                emptySlots_gpu1, firstEmptySlotItr_gpu1 = calcEmptySlots(dest_gpuId)
                emptySlots_gpu0 += max(0, (midEnd - len(gpuIterList[src_gpuId])))*batch_size
                emptySlots_gpu1 += max(0, (midEnd - len(gpuIterList[dest_gpuId]src_gpuId])))*batch_size
                

                # midEnd = int(math.ceil((len(gpuIterList[0])+len(gpuIterList[1]))/2))

                # src_gpuId = 0
                # dest_gpuId = 1
                # emptySlotsInDestGPU = emptySlots_gpu1
                # firstEmptySlotItr_destGPU = firstEmptySlotItr_gpu1
                # if emptySlots_gpu0 > emptySlots_gpu1:
                #     src_gpuId = 1
                #     dest_gpuId = 0
                #     emptySlotsInDestGPU = emptySlots_gpu0
                #     firstEmptySlotItr_destGPU = firstEmptySlotItr_gpu0
                # print ("emptySlotsInDestGPU:", emptySlotsInDestGPU)



                


                # if emptySlotsInDestGPU <= extra_slots_tolerance_reassignment_2*2:
                    # break
                # emptySlotsInDestGPU = calcEmptySlots(dest_gpuId) + max(0, (midEnd - len(gpuIterList[dest_gpuId])))*batch_size
                clusterRepresentatives = self.make_clusters(src_gpuId)
                # for clusterRepresentative in clusterRepresentatives:
                maxBenefitTokenCnt, maxBenefitTokenReqId, prefill_or_decode = findMaxBenefitTokenReq(src_gpuId, dest_gpuId, clusterRepresentatives, emptySlotsInDestGPU + extra_slots_tolerance_reassignment_2, firstEmptySlotItr_destGPU)
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

                    # GPU_processing_through_threading(printAllowed=0)
                itrCnt += 1

        return final_sch, compute_layouts_lists_updated 

    def _create_runners(self):
        replica_resource_mapping = self._get_replica_resource_mapping()

        if not self.is_multi_replica:
            return [BenchmarkRunner(0, self.config, replica_resource_mapping[0])]

        runner_class = ray.remote(num_cpus=1)(BenchmarkRunner)

        runners = []

        set_seeds(self.config.seed)
        # request_generator = RequestGeneratorRegistry.get(
        #     self.config.request_generator_config.get_type(),
        #     self.config.request_generator_config,
        # )
        # requests = request_generator.generate()
        test_seq_3_request = requests[-1]

        for replica_id in range(self.config.num_replicas):
            runners.append(
                runner_class.options(
                    resources={
                        replica_resource_mapping[replica_id][0][0]: 0.01,
                    },
                ).remote(replica_id, self.config, replica_resource_mapping[replica_id])
            )
        # requests[replica_id :len(requests)-2: self.config.num_replicas]

        #TO DO Sudipta: niche ekhon alternately distribute kortesi, later own logic add koro.
        length_aware_sch_res, compute_layouts_list = self.length_aware_scheduling(self.instances, self.requests)
        compute_layouts_list = self.simulate(length_aware_sch_res, compute_layouts_list)
        dist_aware_sch_res, compute_layouts_list = self.distribution_aware_scheduling(compute_layouts_list)
        compute_layouts_list = self.simulate(length_aware_sch_res, compute_layouts_list)
        final_sch, compute_layouts_list = self.selective_distributed(compute_layouts_list)#nicher code guli ei function er vitore chole jabe.

        seq_id_list_runners = []
        seq_category_list_runners = []
        prefill_how_many_in_this_replica_list_runners = []
        decode_how_many_in_this_replica_list_runners = []
        current_replica_id_list_runners = []
        prev_seq_id_list_runners = []
        prev_seq_replica_id_list_runners = []

        for replica_id in range(0, len(runners)):
            seq_id_list_runners.append([])
            seq_category_list_runners.append([])
            prefill_how_many_in_this_replica_list_runners.append([])
            decode_how_many_in_this_replica_list_runners.append([])
            current_replica_id_list_runners.append([])
            prev_seq_id_list_runners.append([])
            prev_seq_replica_id_list_runners.append([])


        for reqId in range(0, len(requests)-1):
            replica_id = find_replica(final_sch, reqId)
            
            seq_category = 0
            prefill_how_many_in_this_replica = -2
            decode_how_many_in_this_replica = -2
            current_replica_id = replica_id
            prev_seq_id = -1
            prev_seq_replica_id = -1

            
            runners[replica_id].append_a_request.remote(requests[reqId])#TO DO Sudipta: append other things here as well, like seqid.
            
            seq_id_list_runners[replica_id].append(reqId)
            seq_category_list_runners[replica_id].append(seq_category)
            prefill_how_many_in_this_replica_list_runners[replica_id].append(prefill_how_many_in_this_replica)
            decode_how_many_in_this_replica_list_runners[replica_id].append(decode_how_many_in_this_replica)
            current_replica_id_list_runners[replica_id].append(current_replica_id)
            prev_seq_id_list_runners[replica_id].append(prev_seq_id)
            prev_seq_replica_id_list_runners[replica_id].append(prev_seq_replica_id)


        
        return runners

    def _create_aggregate_metric_store(self):
        replica_config = ReplicaConfig(
            replica_id=0,  # dummy replica id
            output_dir=self.config.output_dir,
        )
        metrics_store = MetricsStore.get_instance(
            replica_config,
            self.config.model_config,
            self.config.metrics_config,
        )

        if wandb.run is not None:
            wandb.config.update(self.config.to_dict())

        metrics_store.mark_initial_memory_profiling_done()

        return metrics_store

    def run(self):
        if self.is_multi_replica:
            


            ray.get([runner.warmup.remote() for runner in self.runners])

            #testing communication: #TO DO Sudipta: add group_name, remove all print/logger type things since they pass through CPU.
            # _options = {
            #     "world_size": 2,
            #     "ranks": [0, 1],
            #     "backend": "nccl"
            # }
            # col.create_collective_group(self.runners, world_size=2, ranks=[0, 1])
            # list_to_transfer = [1, 2, 3, 4]
            # # tensor_to_transfer = torch.tensor(list_to_transfer)
            # transferred_tensors = ray.get([self.runners[0].do_send.remote(target_rank=1), self.runners[1].do_recv.remote(src_rank=0)])
            
            # self.runners[0].do_send.remote(target_rank=1)
            # self.runners[1].do_recv.remote(src_rank=0)
            
            # logger.info(f"Transferred tensor is: {transferred_tensor}")

            runner_metrics = ray.get([runner.run.remote() for runner in self.runners])

            for runner_metric in runner_metrics:
                self.aggregate_metric_store.merge(runner_metric)

            if wandb.run is not None:
                wandb.config.update(self.config.__dict__)

            # self.aggregate_metric_store.plot()
        else:
            metric_store = self.runners[0].run()
            # metric_store.plot()

        wandb.finish()

class RayColMigrationBackend(MigrationBackendBase):
    def __init__(self, migration_config: MigrationConfig, cache_engine: CacheEngine, local_rank,
                 scheduling_strategy, is_driver_worker, gpu_cache) -> None:
        super().__init__()

        # pylint: disable=C0415
        import cupy

        self.migration_config = migration_config
        self.cache_engine = cache_engine
        self.backend = migration_config.migration_backend
        self.migration_num_layers = min(migration_config.migration_num_layers, self.cache_engine.num_layers)
        self.num_migration_cache_blocks = migration_config.migration_cache_blocks

        self.backend = migration_config.migration_backend
        self.global_world_size = -1
        self.global_rank = -1
        self.group_name = None

        self.local_rank = local_rank
        self.actor = ProxyActor.options(scheduling_strategy=scheduling_strategy).remote()
        self.is_driver_worker = is_driver_worker
        self.gpu_cache = gpu_cache

        self.migration_cache_size = self.cache_engine.block_size * self.cache_engine.num_heads * self.cache_engine.head_size

        if self.backend == 'gloo':
            try_import_gloo()
            self.cache_device = "cpu"
        else:
            self.cache_device = torch.device(f"cuda:{self.local_rank}")

        pin_memory = (self.backend == 'gloo')
        self.dummy_cache = torch.empty(
            size=(self.num_migration_cache_blocks, self.migration_num_layers, 2, self.migration_cache_size),
            dtype=self.cache_engine.dtype,
            device=self.cache_device,
            pin_memory=pin_memory
        )

        self.migration_stream = cupy.cuda.Stream()

    def init_backend(self, group_name, world_size, rank) -> bool:
        @func_set_timeout(self.migration_config.migration_backend_init_timeout)
        def init_group(world_size, rank, backend, group_name):
            col.init_collective_group(world_size, rank, backend, group_name)

        try:
            init_group(world_size, rank, self.backend, group_name)
        except FunctionTimedOut:
            logger.info("create migration backend fail (group_name: {}, world_size: {}, rank: {}, backbend: {})."
                .format(group_name, world_size, rank, self.backend))
            return False

        self.group_name = group_name
        self.global_world_size = world_size
        self.global_rank = rank

        logger.info("create migration backend group successfully (group_name: {}, world_size: {}, rank: {}, backbend: {})."
                    .format(self.group_name, self.global_world_size, self.global_rank, self.backend))
        return True

    def destory_backend(self) -> None:
        if self.group_name is None:
            return

        err_info = None
        try:
            col.destroy_collective_group(self.group_name)
        # pylint: disable=W0703
        except Exception as e:
            err_info = e

        if err_info is not None:
            logger.info("destory migration backend successfully (group_name: {}, backbend: {}), error: {}."
                    .format(self.group_name, self.backend, err_info))
        else:
            logger.info("destory migration backend successfully (group_name: {}, backbend: {})."
                    .format(self.group_name, self.backend))

        self.group_name = None

    def warmup(self) -> bool:
        if self.global_world_size > 1:
            try:
                col.allreduce(self.dummy_cache[0], self.group_name)
            # pylint: disable=W0703
            except Exception as e:
                logger.info("warmup migration backend failed (group_name: {}, world_size: {}, rank: {}, backbend: {}), err: {}."
                    .format(self.group_name, self.global_world_size, self.global_rank, self.backend, e))
                return False

        logger.info("migration backend warmup successfully (group_name: {}, world_size: {}, rank: {}, backbend: {})."
                    .format(self.group_name, self.global_world_size, self.global_rank, self.backend))
        return True

    # Ray.collective is used to construct the gloo and nccl backends. The do_send/do_recv functions will transmit
    # data layer by layer. Take into consideration that col.send/recv are blocking operations.
    def migrate_cache(self, src_handle, dest_handle, src_blocks: List[int], dst_blocks: List[int]) -> None:
        tot_blocks = len(src_blocks)
        src_rank = ray.get(self.actor.exec_method.remote(self.is_driver_worker, src_handle, "get_global_rank"))

        for start_idx in range(0, tot_blocks, self.num_migration_cache_blocks):
            offset = min(self.num_migration_cache_blocks, tot_blocks - start_idx)
            send_blocks = src_blocks[start_idx:start_idx+offset]
            recv_blocks = dst_blocks[start_idx:start_idx+offset]
            self.runners[src_handle].exec_method.remote("do_send", dst_handle, send_blocks)
            self.runners[dst_handle].exec_method.remote("do_recv", src_handle, recv_blocks)

    def do_send(self, dst_handle, blocks: List[int]):
        num_blocks = len(blocks)
        send_cache = self.dummy_cache[:num_blocks].view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)
        src_to_dst = {block_num: idx for idx, block_num in enumerate(blocks)}

        with self.migration_stream: #to handle blocking calls.
            #gpu_cache: List[torch.Tensor]
            for layer_idx in range(self.cache_engine.num_layers):
                cache_idx = layer_idx % self.migration_num_layers
                self.cache_engine.attn_backend.swap_blocks(self.gpu_cache[layer_idx], send_cache[cache_idx], src_to_dst)
                if cache_idx + 1 == self.migration_num_layers or layer_idx + 1 == self.cache_engine.num_layers:
                    # TODO(Sudipta): check the error code if receiving GPU is unresponsive
                    col.send(send_cache, dst_handle, self.group_name)
        self.migration_stream.synchronize()

    def do_send_concat(self, dst_handle, blocks: List[int]):
        num_blocks = len(blocks)
        send_cache = self.dummy_cache[:num_blocks].view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)
        src_to_dst = {block_num: idx for idx, block_num in enumerate(blocks)}

        for layer_idx in range(self.cache_engine.num_layers):
            cache_idx = layer_idx % self.migration_num_layers
            self.concatenated_tensor=torch.cat(self.concatenated_tensor, self.gpu_cache[layer_idx], dim=0)
        with self.migration_stream: #to handle blocking calls.
            #gpu_cache: List[torch.Tensor]
            
                
            self.cache_engine.attn_backend.swap_blocks(self.concatenated_tensor, send_cache[cache_idx], src_to_dst)
                
            # TODO(Sudipta): check the error code if receiving GPU is unresponsive
            col.send(send_cache, dst_handle, self.group_name)
        self.migration_stream.synchronize()

    def do_recv(self, src_handle, blocks: List[int]):
        num_blocks = len(blocks)
        src_to_dst = dict(enumerate(blocks))
        recv_cache = self.dummy_cache[:num_blocks].view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)


        with self.migration_stream:
            for layer_idx in range(self.cache_engine.num_layers):
                cache_idx = layer_idx % self.migration_num_layers
                if cache_idx == 0:
                    col.recv(recv_cache, src_handle, self.group_name)
                self.cache_engine.attn_backend.swap_blocks(recv_cache[cache_idx], self.gpu_cache[layer_idx], src_to_dst)
        self.migration_stream.synchronize()

    def do_recv_concatenate(self, src_handle, blocks: List[int]):
        num_blocks = len(blocks)
        src_to_dst = dict(enumerate(blocks))
        recv_cache = self.dummy_cache[:num_blocks].view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)
 
        with self.migration_stream:
            col.recv(recv_cache, src_handle, self.group_name)

        self.migration_stream.synchronize()

        for layer_idx in range(self.recv_cache):
            cache_idx = layer_idx % self.migration_num_layers
            if cache_idx == 0:
                dummy=0
            self.cache_engine.attn_backend.swap_blocks(recv_cache[cache_idx], self.gpu_cache[layer_idx], src_to_dst)

def get_migration_backend(migration_config: MigrationConfig, cache_engine: CacheEngine, worker_handle_list, scheduling_strategy,
                        is_driver_worker, gpu_cache, worker_rank, local_rank) -> MigrationBackendBase:
    if cache_engine.num_gpu_blocks < migration_config.migration_cache_blocks:
        logger.warning("migration_cache_blocks({}) is larger than num_gpu_blocks({}), reducing it to num_gpu_blocks."
                       .format(migration_config.migration_cache_blocks, cache_engine.num_gpu_blocks))
        migration_config.migration_cache_blocks = cache_engine.num_gpu_blocks

    target_col = None
    backend = migration_config.migration_backend
    if backend in ['nccl', 'gloo']:
        target_col = RayColMigrationBackend(migration_config, cache_engine, local_rank, scheduling_strategy,
                                            is_driver_worker, gpu_cache)
    else:
        target_col = RayRpcMigrationBackend(migration_config, cache_engine, worker_rank, worker_handle_list,
                                            scheduling_strategy, is_driver_worker, gpu_cache)

    return target_col
