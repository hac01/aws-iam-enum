#!/usr/bin/env python3
"""
aws-perm-enum.py — action-aware AWS IAM permission enumerator

Read-only enumerators (enumerate-iam, aws-enumerator) only fire parameterless
List*/Describe*/Get* calls. That leaves two big blind spots:

  1. Action-based permissions that REQUIRE an argument (ssm:StartSession needs
     a target, secretsmanager:GetSecretValue needs a SecretId, ...) are never
     attempted, so they never show up even when granted.
  2. Mutating permissions (ec2:RunInstances, ec2:TerminateInstances, ...) are
     skipped entirely because actually calling them would change state.

This tool closes both gaps without ever changing anything in the account.

How it decides ALLOW vs deny
----------------------------
AWS evaluates authorization BEFORE it validates the argument, so the error
code tells you whether you are allowed:

  * AccessDenied / AccessDeniedException / UnauthorizedOperation   -> DENIED
  * DryRunOperation                                                -> ALLOWED
  * any other error (InvalidInstanceId, ResourceNotFoundException,
    ValidationException, TargetNotConnected, ...) or a success     -> ALLOWED

Three probe styles
-------------------
  read   : parameterless List/Describe/Get — safe, returns real data or AccessDenied
  probe  : call with a throwaway argument that points at nothing — the authz
           check fires before the "no such resource" error
  dryrun : EC2 only — DryRun=True asks AWS to authorize the call and stop;
           it returns DryRunOperation if allowed, UnauthorizedOperation if not

Every probe is side-effect-free. No object is ever created, written, or
deleted. ssm:StartSession is probed against a non-existent instance, and any
session that somehow opens is torn down immediately.

If you hold iam:SimulatePrincipalPolicy you can enumerate without touching the
live APIs at all — but a low-privilege service identity usually does not, which
is why this tool probes.

Usage
-----
  python3 aws-perm-enum.py \
    --access-key AKIA... \
    --secret-key wJalr... \
    --region us-west-2

  # temporary STS creds:
  python3 aws-perm-enum.py --access-key ASIA... --secret-key ... \
    --session-token ... --region us-west-2

  # limit to specific services:
  python3 aws-perm-enum.py --access-key ... --secret-key ... --services ssm,iam,s3
"""

import argparse
import sys

import boto3
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    ParamValidationError,
)

# Error codes that specifically mean "you are not authorized".
DENIED_CODES = {
    "AccessDenied",
    "AccessDeniedException",
    "UnauthorizedOperation",
    "AuthorizationError",
    "NotAuthorized",
    "Client.UnauthorizedOperation",
}

# Error codes that explicitly confirm authorization succeeded.
ALLOWED_CODES = {"DryRunOperation"}

# Bogus identifiers used only to satisfy required parameters.
FAKE_INSTANCE = "i-00000000000000000"
FAKE_AMI = "ami-00000000000000000"
FAKE_NAME = "perm-enum-probe-does-not-exist"


def build_clients(args):
    sess = boto3.session.Session(
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_key,
        aws_session_token=args.session_token,
        region_name=args.region,
    )
    cfg = Config(retries={"max_attempts": 2, "mode": "standard"})
    names = ["sts", "iam", "s3", "ec2", "ssm", "lambda", "dynamodb",
             "secretsmanager", "kms", "sns", "sqs", "ecr", "ecs", "eks",
             "logs", "cloudtrail"]
    return {n: sess.client(n, config=cfg) for n in names}


def catalog(c):
    """
    Return a list of (service, permission, callable, cleanup_or_None).
    `c` is the dict of boto3 clients.
    """
    return [
        # ── STS ──────────────────────────────────────────────────────────
        ("sts", "sts:GetCallerIdentity", lambda: c["sts"].get_caller_identity(), None),
        ("sts", "sts:GetSessionToken",   lambda: c["sts"].get_session_token(), None),

        # ── IAM ──────────────────────────────────────────────────────────
        ("iam", "iam:ListUsers",    lambda: c["iam"].list_users(MaxItems=1), None),
        ("iam", "iam:ListRoles",    lambda: c["iam"].list_roles(MaxItems=1), None),
        ("iam", "iam:ListPolicies", lambda: c["iam"].list_policies(MaxItems=1), None),
        ("iam", "iam:ListGroups",   lambda: c["iam"].list_groups(MaxItems=1), None),
        ("iam", "iam:GetAccountAuthorizationDetails",
                lambda: c["iam"].get_account_authorization_details(MaxItems=1), None),
        ("iam", "iam:GetAccountSummary", lambda: c["iam"].get_account_summary(), None),

        # ── S3 ───────────────────────────────────────────────────────────
        ("s3", "s3:ListAllMyBuckets", lambda: c["s3"].list_buckets(), None),

        # ── EC2 (reads) ──────────────────────────────────────────────────
        ("ec2", "ec2:DescribeInstances",      lambda: c["ec2"].describe_instances(MaxResults=5), None),
        ("ec2", "ec2:DescribeSecurityGroups", lambda: c["ec2"].describe_security_groups(MaxResults=5), None),
        ("ec2", "ec2:DescribeVpcs",           lambda: c["ec2"].describe_vpcs(), None),
        ("ec2", "ec2:DescribeSnapshots",      lambda: c["ec2"].describe_snapshots(OwnerIds=["self"], MaxResults=5), None),

        # ── EC2 (mutating, via DryRun — no side effects) ─────────────────
        ("ec2", "ec2:RunInstances",
                lambda: c["ec2"].run_instances(ImageId=FAKE_AMI, MaxCount=1, MinCount=1, DryRun=True), None),
        ("ec2", "ec2:TerminateInstances",
                lambda: c["ec2"].terminate_instances(InstanceIds=[FAKE_INSTANCE], DryRun=True), None),
        ("ec2", "ec2:CreateKeyPair",
                lambda: c["ec2"].create_key_pair(KeyName=FAKE_NAME, DryRun=True), None),

        # ── SSM ──────────────────────────────────────────────────────────
        ("ssm", "ssm:DescribeInstanceInformation",
                lambda: c["ssm"].describe_instance_information(MaxResults=5), None),
        ("ssm", "ssm:ListCommands", lambda: c["ssm"].list_commands(MaxResults=5), None),
        ("ssm", "ssm:GetConnectionStatus",
                lambda: c["ssm"].get_connection_status(Target=FAKE_INSTANCE), None),
        ("ssm", "ssm:StartSession",
                lambda: c["ssm"].start_session(Target=FAKE_INSTANCE),
                lambda r: c["ssm"].terminate_session(SessionId=r["SessionId"])),

        # ── Secrets Manager ──────────────────────────────────────────────
        ("secretsmanager", "secretsmanager:ListSecrets",
                lambda: c["secretsmanager"].list_secrets(MaxResults=1), None),
        ("secretsmanager", "secretsmanager:GetSecretValue",
                lambda: c["secretsmanager"].get_secret_value(SecretId=FAKE_NAME), None),

        # ── Lambda ───────────────────────────────────────────────────────
        ("lambda", "lambda:ListFunctions", lambda: c["lambda"].list_functions(MaxItems=1), None),
        ("lambda", "lambda:GetFunction",
                lambda: c["lambda"].get_function(FunctionName=FAKE_NAME), None),

        # ── DynamoDB ─────────────────────────────────────────────────────
        ("dynamodb", "dynamodb:ListTables", lambda: c["dynamodb"].list_tables(Limit=1), None),

        # ── KMS ──────────────────────────────────────────────────────────
        ("kms", "kms:ListKeys",    lambda: c["kms"].list_keys(Limit=1), None),
        ("kms", "kms:ListAliases", lambda: c["kms"].list_aliases(Limit=1), None),

        # ── Messaging ────────────────────────────────────────────────────
        ("sns", "sns:ListTopics", lambda: c["sns"].list_topics(), None),
        ("sqs", "sqs:ListQueues", lambda: c["sqs"].list_queues(MaxResults=1), None),

        # ── Containers ───────────────────────────────────────────────────
        ("ecr", "ecr:DescribeRepositories", lambda: c["ecr"].describe_repositories(maxResults=1), None),
        ("ecs", "ecs:ListClusters",         lambda: c["ecs"].list_clusters(maxResults=1), None),
        ("eks", "eks:ListClusters",         lambda: c["eks"].list_clusters(maxResults=1), None),
        ("eks", "eks:DescribeCluster",      lambda: c["eks"].describe_cluster(name=FAKE_NAME), None),

        # ── Logging / audit ──────────────────────────────────────────────
        ("logs", "logs:DescribeLogGroups",      lambda: c["logs"].describe_log_groups(limit=1), None),
        ("cloudtrail", "cloudtrail:DescribeTrails", lambda: c["cloudtrail"].describe_trails(), None),
    ]


def classify(call, cleanup=None):
    """Run one probe. Returns (allowed: bool, detail: str)."""
    try:
        resp = call()
        if cleanup:                       # unexpected success — undo it
            try:
                cleanup(resp)
            except Exception:
                pass
        return True, "call succeeded"
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "Unknown")
        if code in ALLOWED_CODES:
            return True, f"allowed (dry-run authorized)"
        if code in DENIED_CODES:
            return False, code
        return True, f"allowed (failed on argument: {code})"
    except ParamValidationError as e:
        # We built a bad call; cannot conclude. Treat as unknown/skip.
        return None, f"skipped (param error: {e})"
    except EndpointConnectionError as e:
        return None, f"skipped (endpoint error)"


def main():
    p = argparse.ArgumentParser(
        description="Side-effect-free AWS IAM permission enumerator (read + probe + EC2 dry-run)"
    )
    p.add_argument("--access-key", required=True)
    p.add_argument("--secret-key", required=True)
    p.add_argument("--session-token", default=None, help="only for temporary (STS) credentials")
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--services", default=None,
                   help="comma-separated service filter, e.g. ssm,iam,s3 (default: all)")
    args = p.parse_args()

    clients = build_clients(args)

    try:
        ident = clients["sts"].get_caller_identity()
        print(f"[*] Identity : {ident['Arn']}")
        print(f"[*] Account  : {ident['Account']}")
        print(f"[*] Region   : {args.region}\n")
    except ClientError as e:
        print(f"[-] Credentials invalid: {e}")
        sys.exit(1)

    wanted = set(args.services.split(",")) if args.services else None

    allowed = []
    current = None
    for service, perm, call, cleanup in catalog(clients):
        if wanted and service not in wanted:
            continue
        if service != current:
            current = service
            print(f"\n  {service.upper()}")
        ok, detail = classify(call, cleanup)
        if ok is None:
            print(f"    [skip ] {perm:42s} {detail}")
            continue
        mark = "ALLOW" if ok else "deny "
        print(f"    [{mark}] {perm:42s} {detail}")
        if ok:
            allowed.append(perm)

    print("\n" + "=" * 60)
    print(f"[+] {len(allowed)} permission(s) confirmed ALLOWED:")
    for perm in allowed:
        print(f"      {perm}")

    # Flag the high-impact ones.
    HIGH_IMPACT = {
        "ssm:StartSession", "secretsmanager:GetSecretValue",
        "iam:GetAccountAuthorizationDetails", "ec2:RunInstances",
        "ec2:TerminateInstances", "ec2:CreateKeyPair", "lambda:GetFunction",
    }
    hits = [p for p in allowed if p in HIGH_IMPACT]
    if hits:
        print("\n[!] High-impact permissions worth chaining:")
        for perm in hits:
            print(f"      {perm}")
        if "ssm:StartSession" in hits:
            print("      -> open a node shell: "
                  f"aws ssm start-session --target <INSTANCE_ID> --region {args.region}")


if __name__ == "__main__":
    main()
