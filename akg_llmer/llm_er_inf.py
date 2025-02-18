import json
import sys
import os
from typing import Optional

import pandas as pd
import time
import traceback
import pdb
from tqdm import tqdm
import numpy as np

import argparse

from utils import get_labels, serial_entity

# External Dependencies:
import boto3
from botocore.config import Config

from langchain_community.chat_models import BedrockChat
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def get_bedrock_client(assumed_role: Optional[str] = None, region: str = 'us-east-1', runtime: bool = True, external_id: Optional[str] = None, ep_url: Optional[str] = None):
    """
    Create a boto3 client for Amazon Bedrock, with optional configuration overrides 
    """
    target_region = region

    print(
        f"Create new client\n  Using region: {target_region}:external_id={external_id}: ")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        if external_id:
            response = sts.assume_role(
                RoleArn=str(assumed_role),
                RoleSessionName="langchain-llm-1",
                ExternalId=external_id
            )
        else:
            response = sts.assume_role(
                RoleArn=str(assumed_role),
                RoleSessionName="langchain-llm-1",
            )
        print(f"Using role: {assumed_role} ... sts::successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if runtime:
        service_name = 'bedrock-runtime'
    else:
        service_name = 'bedrock'

    if ep_url:
        bedrock_client = session.client(
            service_name=service_name, config=retry_config, endpoint_url=ep_url, **client_kwargs)
    else:
        bedrock_client = session.client(
            service_name=service_name, config=retry_config, **client_kwargs)

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client


def eval_cls(chain_t, seqs, data_path, output_path, cls_prefix=None):
    df_cls = pd.read_csv(data_path, sep='|')
    if cls_prefix is None:
        cls_prefix = f'Do these two records refer to the same real-world entity or can be connected? Simply answer Yes or No, no explanation required.\n '
    seq_l, seq_r = seqs
    y_pred = []
    sta = time.time()
    for i in tqdm(range(len(df_cls))):
        cls_input = f'(1) {seq_l[df_cls.D1.values[i]]}; (2) {seq_r[df_cls.D2.values[i]]}'
        y_pred.append(chain_t.invoke({'question': cls_prefix+cls_input}))
    dt = time.time() - sta
    with open(output_path, 'w') as f:
        json.dump({'cls': y_pred, 'throughput': len(df_cls)/dt}, f)


def eval_rank(chain_t, seqs, rank_label, output_path, num_neg, rank_prefix=None):
    idx_pos, idx_neg = get_labels(path=rank_label)
    seq_l, seq_r = seqs
    if rank_prefix is None:
        rank_prefix = f'Given an anchor record and {num_neg+1} candidate records, rank the candidate records based on whether they refer to the same real-world entity as the anchor record. Return all ranked indices, and please strictly follow the format of "Rank: 2, 3, 1, ...", no explanation required.\n '
        # question_prefix = f'Given an anchor record and {num_neg+1} candidate records, please give the rank of candidate records
        # whether they refer to the same real-world entity as the anchor record. Return all the ranked indices
        # and please strictly follow the format "Rank: 2, 3, 1, ..." with no explanation needed.\n '
    output = []
    sta = time.time()
    for i in tqdm(range(len(idx_pos))):
        rank_input = f'Anchor record: {seq_l[idx_pos[i]]} \n Candidate records: {"; ".join(np.array(seq_r)[idx_neg[i,:num_neg+1]])}.'
        output.append(chain_t.invoke({'question': rank_prefix+rank_input}))
    dt = time.time() - sta

    with open(output_path, 'w') as f:
        json.dump({'rank': output, 'throughput': len(idx_pos)*(num_neg+1)/dt}, f)


def main():
    parser = argparse.ArgumentParser(
        description='LLM ER Serialization Benchmark')
    parser.add_argument('--idx', type=str, default='d10')
    parser.add_argument('--num_neg', type=int, default=10)
    parser.add_argument('--stype', type=str, default='fixed',
                        choices=['fixed', 'random', 'valid', 'plain', 'span', 'pairwise', 'walk'])
    parser.add_argument('--config_data', type=str, default='config_data.json')
    parser.add_argument('--modelId', type=str,
                        default='anthropic.claude-3-sonnet-20240229-v1:0')
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--eval_cls', action='store_true',
                        help='evaluation the model based on classification metrics')
    parser.add_argument('--eval_rank', action='store_true',
                        help='evaluation the model based on ranking metrics')

    sys_argv = sys.argv
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    print(f'Using parameters: {args}')

    bedrock_runtime = get_bedrock_client()

    cl_llm = BedrockChat(
        model_id=args.modelId,
        client=bedrock_runtime,
        # model_kwargs={"max_tokens_to_sample": 100},
        model_kwargs={"temperature": args.temp},
    )
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=cl_llm, verbose=True, memory=memory)

    prompt_t = ChatPromptTemplate.from_messages(
        [("human", "Answer this {question}.")])
    chain_t = prompt_t | cl_llm | StrOutputParser()

    with open(args.config_data, 'r') as f:
        config_data = json.load(f)[args.idx]

    # obtain data configuration
    config_data['name'] = args.idx
    root_path = config_data['root_path']
    dataset = config_data['dataset']
    ground_truth = config_data['ground_truth']
    num_neg = min(config_data['num_neg'], args.num_neg)
    task = ground_truth[2:]
    rank_label = config_data['rank_label']

    # serialization
    seq_l, seq_r = serial_entity(config_data, stype=args.stype)

    if args.eval_cls:
        data_path = f"{root_path}/{dataset}/{ground_truth}.csv"
        output_path = f'./llm_cls_{args.stype}_{args.idx}_{task}.json'
        eval_cls(chain_t, [seq_l, seq_r], data_path, output_path)

    if args.eval_rank:
        output_path = f'./llm_rank_{args.stype}_{args.idx}_{task}.json'
        eval_rank(chain_t, [seq_l, seq_r], rank_label, output_path, num_neg)


if __name__ == "__main__":
    try:
        main()
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
