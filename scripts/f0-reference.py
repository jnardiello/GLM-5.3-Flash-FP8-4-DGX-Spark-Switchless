#!/usr/bin/env python3
"""Capture and verify the private four-rank F0 rollback reference.

All cluster operations in this tool are read-only. ``plan-restore`` writes a private
review document; it never runs any command from that document.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


# This tool is also executed directly from a sealed archive. Importing its sibling
# checker must never add bytecode to that archive and invalidate the exact file set.
sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[1]
REFERENCE_ENV = REPO / "scripts/node/reference/f0-20260912.env"
REFERENCE_MANIFEST = REPO / "scripts/node/reference/f0-20260912.json"
EXPECTED_NCCL_SHA = "1ddc3240396a9b3a1e4fa3e54e129d099261106ce3b9263ac3fdc3e070713bd5"
EXPECTED_NCCL_SIZE = 61_581_280
ARCHIVE_MANIFEST = "SHA256SUMS.json"
MAX_ARCHIVE_FILE = 512 * 1024
SECRET_ASSIGNMENT = re.compile(
    r"(?im)^\s*(?:export\s+)?[A-Za-z0-9_]*(?:PASSWORD|PASSWD|SECRET|PRIVATE_KEY|"
    r"AUTH_TOKEN|ACCESS_TOKEN|API_TOKEN|API_KEY|AUTH_COOKIE)[A-Za-z0-9_]*\s*="
)

_CHECK_SPEC = importlib.util.spec_from_file_location("check_f0", REPO / "scripts/check-f0.py")
if not _CHECK_SPEC or not _CHECK_SPEC.loader:
    raise RuntimeError("cannot load scripts/check-f0.py")
CHECK_F0 = importlib.util.module_from_spec(_CHECK_SPEC)
_CHECK_SPEC.loader.exec_module(CHECK_F0)


REMOTE_NORMALIZER = r'''
def stable_stdout(name, raw):
    if name == "firewall-v4":
        lines=[]
        for line in raw.decode("utf-8","replace").splitlines():
            if line.lstrip().startswith("#"): continue
            line=re.sub(r"^(:\S+\s+\S+\s+)\[\d+:\d+\]$",r"\1[COUNTERS]",line)
            lines.append(line)
        return ("\n".join(lines)+"\n").encode()
    if name in ("ip-link", "ip-address"):
        try: value=json.loads(raw)
        except (ValueError,TypeError): return raw
        if not isinstance(value,list): return raw
        for link in value:
            if not isinstance(link,dict): continue
            info_data=((link.get("linkinfo") or {}).get("info_data") or {})
            if isinstance(info_data,dict): info_data.pop("gc_timer",None)
            if name == "ip-address":
                for address in link.get("addr_info") or []:
                    if isinstance(address,dict):
                        address.pop("preferred_life_time",None)
                        address.pop("valid_life_time",None)
        return json.dumps(value,sort_keys=True,separators=(",",":")).encode()
    return raw
'''


REMOTE_COLLECTOR = r'''
import base64, datetime, hashlib, json, os, pathlib, re, stat, subprocess, sys
''' + REMOTE_NORMALIZER + r'''
p=json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode())
limit=int(p["command_limit"]); total_limit=int(p["total_limit"]); total=0
receipts=[]; problems=[]

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(data): return hashlib.sha256(data).hexdigest()
def run(name, argv, required=False, accepted=(0,), timeout=30, retain=True):
    global total
    started=now()
    try:
        c=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=False)
        rc=c.returncode; out=c.stdout; err=c.stderr; timed=False
    except subprocess.TimeoutExpired as exc:
        rc=None; out=exc.stdout or b""; err=exc.stderr or b""; timed=True
    available=max(0,total_limit-total)
    cap=min(limit,available)
    raw=out+err
    truncated=len(raw)>cap
    if truncated: raw=raw[:cap]
    total+=len(raw)
    unsupported=bool(re.search(br"(?i)(operation not supported|not supported|unknown option|unknown command)",err))
    if timed: status="timeout"
    elif rc in accepted: status="ok"
    elif rc==127 or unsupported or (rc is not None and not pathlib.Path(argv[0]).is_absolute() and not shutil_which(argv[0])):
        status="unsupported"
    else: status="query_error"
    rec={"name":name,"argv":argv,"started_at":started,"finished_at":now(),
         "returncode":rc,"timed_out":timed,"status":status,"truncated":truncated,
         "stdout_sha256":digest(out),"stderr_sha256":digest(err)}
    if retain:
        rec["stdout"]=out[:cap].decode("utf-8","replace")
        rec["stderr"]=err[:max(0,cap-len(out[:cap]))].decode("utf-8","replace")
    receipts.append(rec)
    if truncated: problems.append(name+": output truncated")
    if required and status != "ok": problems.append(name+": "+status)
    elif status in ("timeout","query_error"): problems.append(name+": "+status)
    return out,rc,status

def shutil_which(name):
    for directory in os.environ.get("PATH","").split(os.pathsep):
        if os.access(os.path.join(directory,name),os.X_OK): return True
    return False

def home_path(value):
    if value.startswith("$HOME/"): return str(pathlib.Path.home()/value[6:])
    if value.startswith("~/"): return str(pathlib.Path.home()/value[2:])
    return value

def privileged_record(path):
    code="""import base64,hashlib,json,os,pathlib,stat,sys\npath=pathlib.Path(base64.urlsafe_b64decode(sys.argv[1]).decode()); st=path.lstat(); r={"mode":stat.S_IMODE(st.st_mode),"uid":st.st_uid,"gid":st.st_gid,"size":st.st_size,"type":"symlink" if stat.S_ISLNK(st.st_mode) else "file" if stat.S_ISREG(st.st_mode) else "other"};\nif stat.S_ISLNK(st.st_mode): r["link_target"]=os.readlink(path)\nif stat.S_ISREG(st.st_mode):\n h=hashlib.sha256(); data=path.read_bytes(); h.update(data); r["sha256"]=h.hexdigest(); r["raw"]=base64.b64encode(data).decode()\nprint(json.dumps(r))"""
    encoded=base64.urlsafe_b64encode(str(path).encode()).decode()
    c=subprocess.run(["sudo","-n","python3","-c",code,encoded],capture_output=True,text=True,timeout=30,check=False)
    if c.returncode: raise PermissionError(path)
    return json.loads(c.stdout)

def file_record(value, content=False, required=False, hash_content=True):
    path=pathlib.Path(home_path(value)); rec={"path":str(path),"requested":value}
    try:
        privileged=str(path).startswith(("/etc/", "/boot/grub/"))
        st=None if privileged else path.lstat()
        if privileged:
            detail=privileged_record(path); raw=base64.b64decode(detail.pop("raw","")); rec.update({"present":True,**detail})
            if not hash_content: rec.pop("sha256",None)
            if content and len(raw) <= int(p["file_limit"]):
                if SECRET_ASSIGNMENT.search(raw.decode("utf-8","replace")):
                    rec["content_omitted"]="sensitive assignment"; problems.append(str(path)+": sensitive configuration cannot be archived verbatim")
                else: rec["content_base64"]=base64.b64encode(raw).decode()
            elif content: rec["content_omitted"]="size limit"
            return rec
        rec.update({"present":True,"mode":stat.S_IMODE(st.st_mode),
            "uid":st.st_uid,"gid":st.st_gid,"size":st.st_size,
            "type":"symlink" if stat.S_ISLNK(st.st_mode) else "file" if stat.S_ISREG(st.st_mode) else "other"})
        if stat.S_ISLNK(st.st_mode): rec["link_target"]=os.readlink(path)
        if stat.S_ISREG(st.st_mode) and hash_content:
            h=hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
            rec["sha256"]=h.hexdigest()
            if content and st.st_size <= int(p["file_limit"]):
                raw=path.read_bytes()
                if SECRET_ASSIGNMENT.search(raw.decode("utf-8","replace")):
                    rec["content_omitted"]="sensitive assignment"
                    problems.append(str(path)+": sensitive configuration cannot be archived verbatim")
                else: rec["content_base64"]=base64.b64encode(raw).decode()
            elif content: rec["content_omitted"]="size limit"
    except OSError as exc:
        rec.update({"present":False,"error":type(exc).__name__})
        if required: problems.append(str(path)+": required file missing or unreadable")
    return rec

SECRET_ASSIGNMENT=re.compile(r"(?im)^\s*(?:export\s+)?[A-Za-z0-9_]*(?:PASSWORD|PASSWD|SECRET|PRIVATE_KEY|AUTH_TOKEN|ACCESS_TOKEN|API_TOKEN|API_KEY|AUTH_COOKIE)[A-Za-z0-9_]*\s*=")

def identity():
    boot=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    out,_,status=run("docker-container-inspect",["sudo","-n","docker","inspect",p["container"]],True,timeout=30,retain=False)
    filtered={}
    if status=="ok":
        try:
            obj=json.loads(out)[0]; cfg=obj.get("Config") or {}; host=obj.get("HostConfig") or {}
            all_env=dict(item.split("=",1) for item in cfg.get("Env") or [] if "=" in item)
            allowed={name:all_env[name] for name in p["safe_env"] if name in all_env}
            filtered={"id":obj.get("Id"),"created":obj.get("Created"),
                "state":{key:(obj.get("State") or {}).get(key) for key in ("Status","Running","Pid","StartedAt","Restarting")},
                "restart_count":obj.get("RestartCount"),"image_id":obj.get("Image"),
                "config":{"image":cfg.get("Image"),"entrypoint":cfg.get("Entrypoint"),"cmd":cfg.get("Cmd"),"environment":allowed},
                "host_config":{key:host.get(key) for key in ("NetworkMode","IpcMode","ShmSize","Runtime","DeviceRequests","CapAdd","SecurityOpt","Ulimits")},
                "mounts":sorted([{key:m.get(key) for key in ("Type","Name","Source","Destination","Driver","Mode","RW","Propagation")} for m in obj.get("Mounts") or []],key=lambda m:json.dumps(m,sort_keys=True,separators=(",",":")))}
        except (ValueError,IndexError,KeyError) as exc:
            problems.append("docker-container-inspect: invalid JSON "+type(exc).__name__)
    image={}
    image_ref=(filtered.get("config") or {}).get("image")
    if image_ref:
        raw,_,istatus=run("docker-image-inspect",["sudo","-n","docker","image","inspect",image_ref],True,timeout=30,retain=False)
        if istatus=="ok":
            try:
                io=json.loads(raw)[0]; image={"id":io.get("Id"),"repo_digests":sorted(io.get("RepoDigests") or [])}
            except (ValueError,IndexError): problems.append("docker-image-inspect: invalid JSON")
    return {"boot_id":boot,"container":filtered,"image":image}

before=identity()

base_files=[
    ("$HOME/tp4/cluster.env",True,True),
    ("$HOME/tp4/"+p["launcher"],True,True),
    ("$HOME/tp4/launch-glm53-tp4.sh",True,False),
    ("$HOME/tp4/launch-glm53-tp4-f0-20260906.sh",True,False),
    ("$HOME/tp4/tp4ctl",True,True),
    ("$HOME/tp4/tp4ctl-e04-window-6cb6f07c",True,False),
    ("$HOME/tp4/flusher-unconditional.sh",True,False),
    ("$HOME/tp4/scripts/lib/common.sh",True,False),
    ("$HOME/tp4/scripts/render_chat_template.py",True,False),
    ("$HOME/patches/adaptive_k_scheduler.py",True,True),
    ("$HOME/patches/sparse_attn_indexer_kpool.py",True,True),
    (p["moe_path"],True,True),
    (p["model_dir"]+"/.glm53-fp8-synced",True,True),
    (p["model_dir"]+"/config.json",True,True),
    (p["model_dir"]+"/chat_template.jinja",True,True),
    (p["draft_dir"]+"/config.json",True,True),
    (p["draft_dir"]+"/.cache/huggingface/download/config.json.metadata",True,False),
    (p["draft_dir"]+"/model.safetensors",False,True,False),
    (p["cache_dir"]+"/tp4-chat-template.jinja",True,True),
    (p["nccl_path"],False,True),
    ("/etc/netplan/40-cx7.yaml",True,True),
    ("/etc/default/tp4-fabric-iptables",True,True),
    ("/etc/systemd/system/tp4-autostart.service",True,p["rank"]==0),
    ("/etc/systemd/system/tp4-fabric-iptables.service",True,True),
    ("/etc/systemd/system/tp4-flusher.service",True,False),
    ("/etc/sysctl.d/98-tp4-fabric.conf",True,True),
    ("/etc/sysctl.d/99-tp4-vm.conf",True,True),
    ("/etc/sudoers.d/99-tp4-nopasswd",True,True),
    ("/usr/local/sbin/tp4-fabric-iptables.sh",True,True),
    ("/etc/default/grub",True,True),
    ("/boot/grub/grub.cfg",True,True),
    ("/etc/default/grub.d/zz-tp4-perf.cfg",True,False),
    ("/etc/default/grub.d/.zz-tp4-perf.cfg.reverted",True,False),
]
files=[file_record(*row) for row in base_files]

model=pathlib.Path(home_path(p["model_dir"])); draft=pathlib.Path(home_path(p["draft_dir"]))
availability={"model_shards":None,"model_manifest_present":False,"drafter_weight_files":[]}
try: availability["model_shards"]=len(list(model.glob("model-*-of-*.safetensors")))
except OSError as exc: problems.append("model shard inventory: "+type(exc).__name__)
manifest=pathlib.Path.home()/"tp4/node/model-manifests"/(p["model_revision"]+".json")
availability["model_manifest_present"]=manifest.is_file()
if manifest.is_file(): files.append(file_record(str(manifest),True,True))
try:
    availability["drafter_weight_files"]=[{"name":x.name,"size":x.stat().st_size} for x in sorted(draft.glob("*.safetensors"))]
except OSError as exc: problems.append("drafter inventory: "+type(exc).__name__)

units=("tp4-fabric-iptables","tp4-flusher") if p["rank"]!=0 else ("tp4-autostart","tp4-fabric-iptables","tp4-flusher")
for unit in units:
    run("systemd-show-"+unit,["systemctl","show",unit,"--no-pager","--property=Id,LoadState,ActiveState,SubState,UnitFileState,FragmentPath,DropInPaths,NeedDaemonReload,ExecStart"],True,accepted=(0,3,4))
    run("systemd-cat-"+unit,["systemctl","cat",unit,"--no-pager"],unit!="tp4-flusher",
        accepted=(0,1,3,4) if unit=="tp4-flusher" else (0,))
for receipt in list(receipts):
    if not receipt["name"].startswith("systemd-show-") or receipt["status"]!="ok": continue
    for line in receipt.get("stdout","").splitlines():
        if not line.startswith("DropInPaths="): continue
        for dropin in line.split("=",1)[1].split(): files.append(file_record(dropin,True,True))

commands=[
 ("uname",["uname","-a"],True,(0,)),
 ("os-release",["cat","/etc/os-release"],True,(0,)),
 ("kernel-cmdline",["cat","/proc/cmdline"],True,(0,)),
 ("boot-kernels",["find","/boot","-maxdepth","1","-type","f","-name","vmlinuz-*","-printf","%f\\n"],False,(0,)),
 ("packages",["dpkg-query","-W","-f=${binary:Package}\\t${Version}\\n"],True,(0,)),
 ("package-holds",["apt-mark","showhold"],True,(0,)),
 ("nvidia",["nvidia-smi","--query-gpu=name,uuid,driver_version,temperature.gpu,pstate","--format=csv,noheader,nounits"],True,(0,)),
 ("pci",["lspci","-Dnnk"],True,(0,)),
 ("ip-link",["ip","-details","-json","link","show"],True,(0,)),
 ("ip-address",["ip","-details","-json","address","show"],True,(0,)),
 ("ip-route",["ip","-details","-json","route","show","table","all"],True,(0,)),
 ("ip-neighbor",["ip","-details","-json","neighbor","show"],True,(0,)),
 ("rdma-link",["rdma","link","show"],True,(0,)),
 ("ibdev2netdev",["ibdev2netdev","-v"],True,(0,)),
 ("ibv-devinfo",["ibv_devinfo","-v"],True,(0,)),
 ("networkmanager-devices",["nmcli","-t","-f","DEVICE,TYPE,STATE,CONNECTION","device","status"],False,(0,)),
 ("networkmanager-connections",["nmcli","-t","-f","NAME,UUID,TYPE,DEVICE","connection","show"],False,(0,)),
 ("devlink-dev",["devlink","dev","show"],False,(0,)),
 ("devlink-params",["devlink","dev","param","show"],False,(0,)),
 ("firewall-v4",["sudo","-n","iptables-save"],True,(0,)),
 ("sysctl",["sysctl","kernel.numa_balancing","vm.zone_reclaim_mode","vm.swappiness","net.core.rmem_max","net.core.wmem_max","net.ipv4.conf.all.rp_filter","net.ipv4.tcp_mtu_probing"],True,(0,)),
]
for name,argv,required,accepted in commands: run(name,argv,required,accepted)
devs=[]
for receipt in receipts:
    if receipt["name"]=="devlink-dev" and receipt["status"]=="ok":
        devs=[line.split()[0].removesuffix(":") for line in receipt.get("stdout","").splitlines() if line.split()]
for dev in devs: run("devlink-eswitch-"+dev.replace("/","_"),["sudo","-n","devlink","dev","eswitch","show",dev],False)
for iface in p["interfaces"]:
    run("ethtool-"+iface,["ethtool",iface],True)
    run("ethtool-features-"+iface,["ethtool","-k",iface],True)
    run("ethtool-driver-"+iface,["ethtool","-i",iface],True)
    run("tc-config-qdisc-"+iface,["tc","qdisc","show","dev",iface],False)
    run("tc-config-ingress-"+iface,["tc","filter","show","dev",iface,"ingress"],False)
    run("tc-config-egress-"+iface,["tc","filter","show","dev",iface,"egress"],False)
    run("tc-qdisc-"+iface,["tc","-s","qdisc","show","dev",iface],False)
    run("tc-filter-"+iface,["tc","-s","filter","show","dev",iface],False)
    run("tc-ingress-"+iface,["tc","-s","filter","show","dev",iface,"ingress"],False)
    run("tc-egress-"+iface,["tc","-s","filter","show","dev",iface,"egress"],False)

# Prove which NCCL bytes are mapped by a process inside the serving container.
loaded={"verified":False,"mapped_paths":[],"sha256":None,"size":None}
top,_,top_status=run("docker-top",["sudo","-n","docker","top",p["container"],"-eo","pid"],True)
if top_status=="ok":
    for line in top.decode("utf-8","replace").splitlines()[1:]:
        if not line.strip().isdigit(): continue
        pid=line.strip()
        maps,_,mstatus=run("proc-maps-"+pid,["sudo","-n","cat","/proc/"+pid+"/maps"],False,retain=False)
        if mstatus!="ok": continue
        paths=sorted(set(row.split()[-1] for row in maps.decode("utf-8","replace").splitlines() if "libnccl.so" in row and row.split()))
        if not paths: continue
        loaded["pid"]=int(pid); loaded["mapped_paths"]=paths
        mapped=next((path for path in paths if path.startswith("/") and not path.endswith(" (deleted)")),"")
        candidate="/proc/"+pid+"/root"+mapped
        try:
            h=hashlib.sha256(); size=0
            with open(candidate,"rb") as f:
                for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk); size+=len(chunk)
            loaded.update({"verified":True,"sha256":h.hexdigest(),"size":size,"proof_path":candidate})
        except OSError:
            raw,_,sstatus=run("loaded-nccl-sha",["sudo","-n","sha256sum",candidate],False)
            if sstatus=="ok" and raw.split():
                size_raw,_,zstatus=run("loaded-nccl-size",["sudo","-n","stat","-c","%s",candidate],False)
                size=int(size_raw.strip()) if zstatus=="ok" and size_raw.strip().isdigit() else None
                loaded.update({"verified":True,"sha256":raw.split()[0].decode(),"proof_path":candidate,"size":size})
        break
if not loaded["verified"]: problems.append("loaded NCCL byte identity is unverified")

after=identity()
identity_equal=(before==after)
if not identity_equal: problems.append("runtime identity changed during capture")
stable_commands=("uname","os-release","kernel-cmdline","boot-kernels","packages","package-holds",
 "pci","ip-link","ip-address","ip-route","rdma-link","ibdev2netdev","ibv-devinfo",
 "networkmanager-devices","networkmanager-connections","devlink-dev","devlink-params",
 "firewall-v4","sysctl")
managed_state={r["name"]:{k:r[k] for k in ("status","returncode","timed_out","truncated","stdout_sha256","stderr_sha256")} for r in receipts if r["name"].startswith("systemd-")}
host_state={}
for r in receipts:
    if r["name"] not in stable_commands and not r["name"].startswith(("ethtool-","devlink-eswitch-","tc-config-")): continue
    item={k:r[k] for k in ("status","returncode","timed_out","truncated","stdout_sha256","stderr_sha256")}
    if r["name"] in ("firewall-v4","ip-link","ip-address") and r["status"]=="ok":
        item["stdout_sha256"]=digest(stable_stdout(r["name"],r.get("stdout","").encode()))
        item["normalization"]="volatile-runtime-fields-v1"
    host_state[r["name"]]=item
nvidia_identity=[]
for r in receipts:
    if r["name"]=="nvidia" and r["status"]=="ok":
        nvidia_identity=[[field.strip() for field in line.split(",")[:3]] for line in r.get("stdout","").splitlines()]
strict={"identity":after,"files":[{k:v for k,v in f.items() if k not in ("content_base64",)} for f in files],"model_availability":availability,"loaded_nccl":loaded,"managed_state":managed_state,"host_state":host_state,"nvidia_identity":nvidia_identity}
runtime_signature=hashlib.sha256(json.dumps(strict,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print(json.dumps({"schema":1,"rank":p["rank"],"captured_at":now(),"identity_before":before,
 "identity_after":after,"identity_stable":identity_equal,"runtime_signature":runtime_signature,
 "files":files,"model_availability":availability,"loaded_nccl":loaded,"managed_state":managed_state,"host_state":host_state,"nvidia_identity":nvidia_identity,"commands":receipts,
 "problems":problems},sort_keys=True))
'''


class ReferenceError(RuntimeError):
    pass


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def outside_repo(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == REPO or REPO in resolved.parents:
        raise ReferenceError("private archive/report path must be outside the checkout")
    return resolved


def write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def write_json(path: Path, value: Any) -> None:
    write_private(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def secure_tree(root: Path) -> None:
    for current, dirs, files in os.walk(root, followlinks=False):
        Path(current).chmod(0o700)
        for name in dirs:
            path = Path(current) / name
            if not path.is_symlink(): path.chmod(0o700)
        for name in files:
            path = Path(current) / name
            if not path.is_symlink(): path.chmod(0o600)


def copy_private_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ReferenceError(f"private input directory is missing: {source}")
    links = [path for path in source.rglob("*") if path.is_symlink()]
    if links:
        raise ReferenceError("private input contains a symlink; preserve link facts as JSON metadata")
    shutil.copytree(source, destination)
    secure_tree(destination)


def source_paths(repo: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=repo, capture_output=True, check=True,
    )
    return [item.decode("utf-8", "surrogateescape") for item in result.stdout.split(b"\0") if item]


def copy_source(repo: Path, destination: Path) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, mode=0o700)
    records = []
    for relative in source_paths(repo):
        source = repo / relative; target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        st = source.lstat()
        record = {"path": relative, "mode": stat.S_IMODE(st.st_mode),
                  "uid": st.st_uid, "gid": st.st_gid,
                  "type": "symlink" if source.is_symlink() else "file"}
        if source.is_symlink():
            raise ReferenceError("repository source contains a symlink; record it without archiving an active link")
        elif source.is_file():
            shutil.copyfile(source, target); target.chmod(0o600)
            record.update({"size": st.st_size, "sha256": sha256_file(source)})
        else:
            raise ReferenceError(f"unsupported source entry: {relative}")
        records.append(record)
    secure_tree(destination)
    return records


def safe_config(path: Path) -> bytes:
    data = path.read_bytes()
    text = data.decode("utf-8", "replace")
    if SECRET_ASSIGNMENT.search(text) or re.search(r"(?i)https?://[^/\s:@]+:[^/\s@]+@", text):
        raise ReferenceError(f"refusing to archive credential-like content from {path.name}")
    return data


def prechange_ownership(source_snapshot: Path) -> dict[str, Any]:
    manifest = json.loads((source_snapshot / "manifest.json").read_text())
    records = []
    for item in manifest.get("entries") or []:
        live = REPO / item["path"]
        if not live.exists() and not live.is_symlink():
            records.append({"path": item["path"], "status": "missing_after_snapshot"})
            continue
        st = live.lstat()
        records.append({"path": item["path"], "uid": st.st_uid, "gid": st.st_gid,
                        "sampled_after_content_snapshot": True})
    return {"schema": 1, "captured_at": utcnow(),
            "provenance": "Ownership sampled from the same checkout after the pre-change content snapshot; original content, modes and links come from that snapshot manifest.",
            "entries": records}


def operational_readiness(evidence_dirs: list[Path], ranks: list[dict[str, Any]]) -> dict[str, Any]:
    reports = []
    for directory in evidence_dirs:
        for path in directory.rglob("report.json"):
            try:
                value = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if value.get("status") in ("PASS", "FAIL") and isinstance(value.get("problems"), list):
                reports.append({"source": path.name, "status": value["status"],
                                "reasons": [str(item) for item in value["problems"]]})
    reload_units = []
    for rank, receipt in enumerate(ranks):
        for command in receipt.get("commands") or []:
            if command.get("name", "").startswith("systemd-show-") and "NeedDaemonReload=yes" in command.get("stdout", ""):
                reload_units.append(f"rank {rank}: {command['name'].removeprefix('systemd-show-')} NeedDaemonReload=yes")
    reasons = [reason for report in reports if report["status"] == "FAIL" for reason in report["reasons"]]
    reasons.extend(reload_units)
    if any(report["status"] == "FAIL" for report in reports) or reload_units: status = "FAIL"
    elif any(report["status"] == "PASS" for report in reports): status = "PASS"
    else: status = "UNKNOWN"
    return {"status": status, "reasons": sorted(set(reasons)),
            "source_report_count": len(reports),
            "meaning": "Operational check result is separate from archive completeness and identity comparison."}


def load_recipe(timeout: float) -> dict[str, Any]:
    recipe, _ = CHECK_F0.load_recipe(timeout)
    shell = r'''
set -euo pipefail
repo=$1
TP4_LOG_TAG='[f0-reference]'
. "$repo/scripts/lib/common.sh"
. "$repo/cluster.env"
tp4_check_env "$repo"
tp4_load_env "$repo" --require --overlay
emit() { printf '%s\0%s\0' "$1" "$2"; }
emit launcher "$LAUNCHER"; emit nccl_dir "$NCCL_DIR"; emit patch_file "$PATCH_FILE"
emit cache_dir "$CACHE_DIR"; emit block_size "$BLOCK_SIZE"; emit gpu_mem_util "$GPU_MEM_UTIL"
emit relay_dest "$RELAY_DEST"; emit node_hostnames "${NODE_HOSTNAMES-}"
for i in 0 1 2 3; do
  emit "netplan_renderer_$i" "$(tp4_resolve_rank_value "$i" NETPLAN_RENDERER NETPLAN_RENDERER_BY_RANK "$TP4_DEFAULT_NETPLAN_RENDERER")"
done
'''
    result = subprocess.run(["bash", "-c", shell, "f0-reference", str(REPO)],
                            capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        raise ReferenceError("effective configuration is invalid")
    fields = result.stdout.split(b"\0")
    if fields and not fields[-1]: fields.pop()
    if len(fields) % 2: raise ReferenceError("extended recipe output is malformed")
    recipe.update({fields[i].decode(): fields[i + 1].decode() for i in range(0, len(fields), 2)})
    manifest = json.loads(REFERENCE_MANIFEST.read_text())
    recipe["reference_manifest"] = manifest
    return recipe


def ssh_argv(host: str, timeout: float) -> list[str]:
    connect = max(1, min(15, math.ceil(timeout)))
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", f"ConnectTimeout={connect}", "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2", host]


def resolved_site_env(recipe: dict[str, Any]) -> bytes:
    scalar_keys = ("nodes", "node_hostnames", "tp4_hosts", "mgmt_ips", "master_ip", "master_port",
                   "api_port", "container", "relay_dest", "fabric_prefix_re")
    lines = ["# Private, resolved F0 site identity captured before E07.\n"]
    names = {"nodes": "NODES", "node_hostnames": "NODE_HOSTNAMES",
             "tp4_hosts": "TP4_HOSTS", "mgmt_ips": "MGMT_IPS",
             "master_ip": "MASTER_IP", "master_port": "MASTER_PORT", "api_port": "API_PORT",
             "container": "CONTAINER", "relay_dest": "RELAY_DEST",
             "fabric_prefix_re": "FABRIC_PREFIX_RE"}
    for key in scalar_keys:
        if key in recipe: lines.append(f"{names[key]}={shlex.quote(str(recipe[key]))}\n")
    def array(name: str, values: list[str]) -> None:
        lines.append(name + "=(" + " ".join(shlex.quote(value) for value in values) + ")\n")
    array("FABRIC_TARGETS", [recipe[f"fabric_target_{rank}"] for rank in range(4)])
    array("MGMT_IF_BY_RANK", [recipe[f"mgmt_if_{rank}"] for rank in range(4)])
    array("FABRIC_IFACES_BY_RANK", [recipe[f"fabric_ifaces_{rank}"] for rank in range(4)])
    array("NCCL_IB_HCA_BY_RANK", [recipe[f"hca_{rank}"] for rank in range(4)])
    array("NETPLAN_RENDERER_BY_RANK", [recipe[f"netplan_renderer_{rank}"] for rank in range(4)])
    return "".join(lines).encode()


def complete_f0_cluster(site: bytes, reference: bytes, resolved: bytes) -> bytes:
    return (site.rstrip() + b"\n\n# Frozen F0 non-site runtime values.\n" + reference.rstrip()
            + b"\n\n# Resolved site values: these override scalar/default inference.\n"
            + resolved.rstrip() + b"\n")


def rank_payload(rank: int, recipe: dict[str, Any], command_limit: int,
                 total_limit: int) -> dict[str, Any]:
    extra = shlex.split(recipe["extra_docker_env"])
    moe_source = next((extra[i + 1].split(":", 1)[0] for i, value in enumerate(extra[:-1])
                       if value == "-v" and "moe-configs" in extra[i + 1]), "")
    safe_env = [
        "LD_PRELOAD", "VLLM_NCCL_SO_PATH", "NCCL_SKIP_TREE_CONNECT", "NCCL_NET",
        "NCCL_IB_DISABLE", "NCCL_IB_SUBNET_PREFIX_LEN", "NCCL_IB_SUBNET_AWARE_ROUTING",
        "NCCL_ALGO", "NCCL_PROTO", "NCCL_P2P_LEVEL", "NCCL_MIN_NCHANNELS",
        "NCCL_MAX_NCHANNELS", "NCCL_CROSS_NIC", "NCCL_CUMEM_ENABLE", "NCCL_NVLS_ENABLE",
        "NCCL_IGNORE_CPU_AFFINITY", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
        "TP_SOCKET_IFNAME", "MN_IF_NAME", "VLLM_HOST_IP", "VLLM_ENGINE_READY_TIMEOUT_S",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING", "PYTORCH_CUDA_ALLOC_CONF", "TORCH_CUDA_ARCH_LIST",
        "FLASHINFER_CUDA_ARCH_LIST", "FLASHINFER_DISABLE_VERSION_CHECK", "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE", "PYTHONUNBUFFERED", "HF_HOME", "XDG_CACHE_HOME",
        "VLLM_CACHE_ROOT", "PYTHONPATH", "VLLM_ADAPTIVE_K_ENABLE", "VLLM_ADAPTIVE_K_LO",
        "VLLM_ADAPTIVE_K_HI", "VLLM_ADAPTIVE_K_MODE", "VLLM_ADAPTIVE_K_SEED",
        "VLLM_ADAPTIVE_K_DOWN", "VLLM_ADAPTIVE_K_UP", "VLLM_ADAPTIVE_K_ALPHA",
        "VLLM_ADAPTIVE_K_SIGNAL", "VLLM_ADAPTIVE_K_LOG_EVERY", "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX", "NCCL_IB_ROCE_VERSION_NUM", "NCCL_IB_ADDR_FAMILY",
    ]
    return {
        "rank": rank, "container": recipe["container"], "launcher": recipe["launcher"],
        "model_dir": recipe["model_dir"], "draft_dir": recipe["draft_dir"],
        "cache_dir": recipe["cache_dir"],
        "model_revision": recipe["model_rev"], "nccl_path": recipe["nccl_dir"] + "/libnccl.so.2",
        "nccl_size": EXPECTED_NCCL_SIZE, "moe_path": moe_source,
        "interfaces": recipe[f"fabric_ifaces_{rank}"].split(), "safe_env": safe_env,
        "command_limit": command_limit, "total_limit": total_limit, "file_limit": MAX_ARCHIVE_FILE,
    }


def collect_rank(rank: int, host: str, recipe: dict[str, Any], timeout: float,
                 command_limit: int, total_limit: int) -> dict[str, Any]:
    payload = base64.urlsafe_b64encode(json.dumps(
        rank_payload(rank, recipe, command_limit, total_limit)).encode()).decode()
    argv = ssh_argv(host, timeout) + ["python3", "-", payload]
    try:
        result = subprocess.run(argv, input=REMOTE_COLLECTOR, text=True, capture_output=True,
                                timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"rank": rank, "capture_status": "timeout", "problems": ["rank capture timeout"]}
    if result.returncode:
        return {"rank": rank, "capture_status": "ssh_error", "returncode": result.returncode,
                "problems": ["rank capture SSH failure"]}
    if len(result.stdout.encode()) > total_limit + 1024 * 1024:
        return {"rank": rank, "capture_status": "oversize", "problems": ["rank response exceeded bound"]}
    try: value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"rank": rank, "capture_status": "invalid", "problems": ["invalid rank response"]}
    value["capture_status"] = "complete" if not value.get("problems") else "incomplete"
    return value


def copy_remote_file(host: str, absolute_path: str, destination: Path, timeout: float) -> dict[str, Any]:
    code = ("import base64,pathlib,sys\n"
            "p=pathlib.Path(base64.urlsafe_b64decode(sys.argv[1]).decode())\n"
            "sys.stdout.buffer.write(p.read_bytes())\n")
    encoded = base64.urlsafe_b64encode(absolute_path.encode()).decode()
    argv = ssh_argv(host, timeout) + ["python3", "-", encoded]
    destination.parent.mkdir(parents=True, exist_ok=True); destination.parent.chmod(0o700)
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    started = utcnow()
    try:
        with os.fdopen(fd, "wb") as stream:
            result = subprocess.run(argv, input=code.encode(), stdout=stream, stderr=subprocess.PIPE,
                                    timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        destination.unlink(missing_ok=True)
        return {"started_at": started, "finished_at": utcnow(), "status": "timeout"}
    record = {"started_at": started, "finished_at": utcnow(), "returncode": result.returncode,
              "status": "ok" if result.returncode == 0 else "ssh_error"}
    if result.returncode:
        destination.unlink(missing_ok=True); return record
    record.update({"size": destination.stat().st_size, "sha256": sha256_file(destination)})
    return record


def materialize_remote_files(rank_dir: Path, rank: dict[str, Any]) -> None:
    target = rank_dir / "files"; target.mkdir(parents=True, mode=0o700)
    for index, item in enumerate(rank.get("files") or []):
        content = item.pop("content_base64", None)
        if content is None: continue
        name = f"{index:02d}-{Path(item['path']).name}"
        write_private(target / name, base64.b64decode(content))
        item["archive_path"] = str(Path("ranks") / rank_dir.name / "files" / name)


def render_private(source: Path, site_config: Path, destination: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="f0-render-", dir="/private/tmp") as temp:
        work = Path(temp) / "repo"
        shutil.copytree(source, work, symlinks=True)
        shutil.copyfile(site_config, work / "cluster.env")
        result = subprocess.run(["bash", "scripts/render-netplan.sh", "--write"], cwd=work,
                                capture_output=True, text=True, timeout=60, check=False)
        if result.returncode:
            return {"status": "FAIL", "returncode": result.returncode}
        generated = []
        for path in sorted((work / "scripts/node/etc").glob("*/*")):
            if not path.is_file(): continue
            if path.parent.name in ("common", "default", "local"): continue
            relative = path.relative_to(work / "scripts/node/etc")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target); target.chmod(0o600)
            generated.append({"path": str(relative), "sha256": sha256_file(target), "size": target.stat().st_size})
        return {"status": "PASS", "generated": generated}


def archive_hashes(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.name == ARCHIVE_MANIFEST: continue
        if path.is_symlink():
            target = os.readlink(path)
            records.append({"path": str(path.relative_to(root)), "type": "symlink",
                            "link_target": target,
                            "sha256": hashlib.sha256(target.encode("utf-8", "surrogateescape")).hexdigest()})
        elif path.is_file():
            records.append({"path": str(path.relative_to(root)), "type": "file",
                            "size": path.stat().st_size, "sha256": sha256_file(path),
                            "archive_mode": "0600"})
    return records


def public_reference_problems() -> list[str]:
    problems = []
    manifest = json.loads(REFERENCE_MANIFEST.read_text())
    for item in manifest["artifacts"]:
        path = REPO / item["path"]
        if not path.is_file(): problems.append("missing public artifact: " + item["path"])
        elif sha256_file(path) != item["sha256"]: problems.append("public artifact hash: " + item["path"])
    baseline = REPO / manifest["baseline"]["path"]
    if not baseline.is_file() or sha256_file(baseline) != manifest["baseline"]["sha256"]:
        problems.append("frozen baseline hash")
    return problems


def capture(args: argparse.Namespace, collector: Callable = collect_rank) -> int:
    archive = outside_repo(args.archive)
    if archive.exists():
        print("F0 REFERENCE CAPTURE FAIL archive already exists")
        return 1
    archive.mkdir(parents=True, mode=0o700)
    status: dict[str, Any] = {"schema": 1, "reference_id": "f0-20260912",
                              "started_at": utcnow(), "status": "INCOMPLETE", "problems": []}
    try:
        problems = public_reference_problems(); recipe = load_recipe(args.timeout)
        # This tool captures the historic F0 reference, never the current default.
        problems.extend(CHECK_F0.recipe_problems(recipe, CHECK_F0.expected_f0(REPO / "docs/baseline-f0.json")))
        site_bytes = safe_config(REPO / "cluster.env")
        resolved_bytes = resolved_site_env(recipe)
        write_private(archive / "private/site/cluster.env", site_bytes)
        overlay = os.environ.get("TP4_ENV")
        if overlay:
            overlay_path = REPO / overlay
            if not overlay_path.is_file(): raise ReferenceError("TP4_ENV file is missing")
            write_private(archive / "private/site/observed-overlay.env", safe_config(overlay_path))
        private_recipe = {key: value for key, value in recipe.items() if key != "reference_manifest"}
        private_recipe["tp4_env"] = overlay
        write_json(archive / "private/site/effective-recipe.json", private_recipe)
        write_private(archive / "private/site/f0-site-resolved.env", resolved_bytes)
        write_private(archive / "private/site/f0-cluster-resolved.env",
                      complete_f0_cluster(site_bytes, REFERENCE_ENV.read_bytes(), resolved_bytes))
        prechange = outside_repo(args.prechange_source)
        write_json(archive / "source/pre-change-ownership.json", prechange_ownership(prechange))
        copy_private_tree(prechange, archive / "source/pre-change")
        final_records = copy_source(REPO, archive / "source/completed-iac")
        write_json(archive / "source/completed-iac-manifest.json", final_records)
        evidence_dirs = [outside_repo(evidence) for evidence in args.evidence_dir]
        for evidence in evidence_dirs:
            copy_private_tree(evidence, archive / "evidence" / evidence.name)
        render = render_private(archive / "source/completed-iac",
                                archive / "private/site/f0-cluster-resolved.env", archive / "rendered")
        write_json(archive / "rendered/render.json", render)
        if render["status"] != "PASS": problems.append("private netplan render failed")
        hosts = recipe["hosts"].split()
        if len(hosts) != 4: raise ReferenceError("effective recipe does not resolve four ranks")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(collector, rank, host, recipe, args.timeout,
                                   args.command_limit, args.total_limit)
                       for rank, host in enumerate(hosts)]
            ranks = [future.result() for future in futures]
        for rank, value in enumerate(ranks):
            rank_dir = archive / f"ranks/rank-{rank}"; rank_dir.mkdir(parents=True, mode=0o700)
            materialize_remote_files(rank_dir, value)
            write_json(rank_dir / "receipt.json", value)
            problems.extend(f"rank {rank}: {item}" for item in value.get("problems") or [])
            if value.get("capture_status") != "complete": problems.append(f"rank {rank}: capture incomplete")
        rank0_cluster = next((item for item in ranks[0].get("files") or []
                              if item.get("requested") == "$HOME/tp4/cluster.env"), {})
        rank0_path = Path(rank0_cluster.get("path", ""))
        if len(rank0_path.parts) < 4 or rank0_path.name != "cluster.env":
            problems.append("rank 0: cannot render private autostart controller path")
        else:
            deploy_home = rank0_path.parent.parent
            template = (REPO / "scripts/node/reference/tp4-autostart-f0-20260912.conf.example").read_text()
            addresses = recipe["mgmt_ips"].split()
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", deploy_home.name) or len(addresses) != 4:
                raise ReferenceError("rank 0: unsafe autostart render identity")
            mesh = " ".join(f"{deploy_home.name}@{address}" for address in addresses)
            rendered_dropin = template.replace("<USER>", deploy_home.name).replace("<TP4_HOSTS>", mesh)
            if "<USER>" in rendered_dropin or "<TP4_HOSTS>" in rendered_dropin:
                problems.append("rank 0: autostart template render token remains")
            else:
                write_private(archive / "private/prepared/tp4-autostart.service.d/20-f0-reference.conf",
                              rendered_dropin.encode())
        first_nccl = next((item for item in ranks[0].get("files") or []
                           if item.get("requested") == recipe["nccl_dir"] + "/libnccl.so.2"), None)
        if not first_nccl or not first_nccl.get("path"):
            problems.append("rank 0: NCCL path was not captured")
            nccl_copy = {"status": "not_attempted"}
        else:
            nccl_copy = copy_remote_file(hosts[0], first_nccl["path"],
                                         archive / "artifacts/libnccl.so.2", args.nccl_timeout)
            if (nccl_copy.get("sha256") != EXPECTED_NCCL_SHA or
                    nccl_copy.get("size") != EXPECTED_NCCL_SIZE):
                problems.append("exact NCCL archive copy is missing or has the wrong identity")
        write_json(archive / "artifacts/nccl-copy.json", nccl_copy)
        for rank, value in enumerate(ranks):
            static = next((item for item in value.get("files") or []
                           if item.get("requested") == recipe["nccl_dir"] + "/libnccl.so.2"), {})
            if static.get("sha256") != EXPECTED_NCCL_SHA or static.get("size") != EXPECTED_NCCL_SIZE:
                problems.append(f"rank {rank}: mounted NCCL file identity")
            loaded = value.get("loaded_nccl") or {}
            if loaded.get("sha256") != EXPECTED_NCCL_SHA or loaded.get("size") != EXPECTED_NCCL_SIZE:
                problems.append(f"rank {rank}: loaded NCCL byte identity")
        readiness = operational_readiness(evidence_dirs, ranks)
        status.update({"finished_at": utcnow(), "status": "COMPLETE" if not problems else "INCOMPLETE",
                       "problems": sorted(set(problems)), "rank_count": len(ranks),
                       "operational_readiness": readiness,
                       "observed_tp4_env": overlay,
                       "observed_autostart_selects_reference": False,
                       "prepared_autostart_overlay": "scripts/node/reference/f0-20260912.env",
                       "runtime_signatures": [item.get("runtime_signature") for item in ranks]})
    except Exception as exc:
        status["finished_at"] = utcnow()
        status["problems"].append(f"{type(exc).__name__}: {str(exc)[:200]}")
    write_json(archive / "archive.json", status)
    secure_tree(archive)
    write_json(archive / ARCHIVE_MANIFEST,
               {"schema": 1, "created_at": utcnow(), "entries": archive_hashes(archive)})
    secure_tree(archive)
    readiness_status = (status.get("operational_readiness") or {}).get("status", "UNKNOWN")
    print(f"F0 REFERENCE CAPTURE {status['status']} readiness={readiness_status} report={archive / 'archive.json'}")
    return 0 if status["status"] == "COMPLETE" else 1


def validate_archive(archive: Path, *, rerender: bool = True) -> list[str]:
    problems = []
    required = ["archive.json", ARCHIVE_MANIFEST, "private/site/cluster.env",
                "private/site/effective-recipe.json", "private/site/f0-site-resolved.env",
                "private/site/f0-cluster-resolved.env",
                "private/prepared/tp4-autostart.service.d/20-f0-reference.conf",
                "source/pre-change/manifest.json", "source/pre-change/git-status.txt",
                "source/pre-change-ownership.json", "source/completed-iac-manifest.json",
                "artifacts/libnccl.so.2", "artifacts/nccl-copy.json", "rendered/render.json",
                *[f"ranks/rank-{rank}/receipt.json" for rank in range(4)]]
    for relative in required:
        if not (archive / relative).is_file(): problems.append("missing archive file: " + relative)
    if problems: return problems
    if stat.S_IMODE(archive.stat().st_mode) != 0o700: problems.append("archive directory mode is not 0700")
    for path in archive.rglob("*"):
        if path.is_symlink(): problems.append("active archive symlink: " + str(path.relative_to(archive)))
        elif path.is_dir() and stat.S_IMODE(path.stat().st_mode) != 0o700:
            problems.append("private directory mode: " + str(path.relative_to(archive)))
    manifest = json.loads((archive / ARCHIVE_MANIFEST).read_text())
    entries = manifest.get("entries")
    if manifest.get("schema") != 1 or not isinstance(entries, list):
        return problems + ["invalid SHA manifest schema"]
    expected = {}
    for item in entries:
        relative = item.get("path") if isinstance(item, dict) else None
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts):
            problems.append("unsafe SHA manifest path")
            continue
        if relative in expected: problems.append("duplicate SHA manifest path: " + relative)
        expected[relative] = item
    actual = {str(path.relative_to(archive)) for path in archive.rglob("*")
              if (path.is_file() or path.is_symlink()) and path.name != ARCHIVE_MANIFEST}
    if actual != set(expected): problems.append("archive file set differs from SHA manifest")
    for relative, item in expected.items():
        path = archive / relative
        if item.get("type") != "file": problems.append("unsupported archive entry type: " + relative); continue
        if not path.is_file(): continue
        if stat.S_IMODE(path.stat().st_mode) != 0o600: problems.append("private file mode: " + relative)
        if path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            problems.append("archive hash: " + relative)
    state = json.loads((archive / "archive.json").read_text())
    if state.get("status") != "COMPLETE": problems.append("capture status is not COMPLETE")
    if state.get("rank_count") != 4 or len(state.get("runtime_signatures") or []) != 4 \
            or any(not value for value in state.get("runtime_signatures") or []):
        problems.append("capture does not contain four runtime signatures")
    readiness = state.get("operational_readiness") or {}
    if readiness.get("status") not in ("PASS", "FAIL", "UNKNOWN") or not isinstance(readiness.get("reasons"), list):
        problems.append("operational readiness status is missing")
    for rank in range(4):
        receipt = json.loads((archive / f"ranks/rank-{rank}/receipt.json").read_text())
        if (receipt.get("rank") != rank or receipt.get("capture_status") != "complete"
                or not receipt.get("identity_stable") or receipt.get("problems")):
            problems.append(f"rank {rank}: receipt is incomplete")
    completed = json.loads((archive / "source/completed-iac-manifest.json").read_text())
    if not isinstance(completed, list) or not completed: problems.append("completed IaC manifest is empty")
    else:
        for item in completed:
            if not all(key in item for key in ("path", "mode", "uid", "gid", "type")):
                problems.append("completed IaC metadata is incomplete"); break
            source_path = archive / "source/completed-iac" / item["path"]
            if item["type"] != "file" or not source_path.is_file():
                problems.append("completed IaC entry is unavailable: " + item["path"]); break
            if source_path.stat().st_size != item["size"] or sha256_file(source_path) != item["sha256"]:
                problems.append("completed IaC entry hash: " + item["path"]); break
    prechange = json.loads((archive / "source/pre-change/manifest.json").read_text())
    if (prechange.get("entry_count") != len(prechange.get("entries") or [])
            or not prechange.get("entries")):
        problems.append("pre-change source manifest is incomplete")
    else:
        for item in prechange["entries"]:
            relative = Path(item.get("path", ""))
            path = archive / "source/pre-change/source" / relative
            if (relative.is_absolute() or ".." in relative.parts or item.get("type") != "file"
                    or not path.is_file() or path.stat().st_size != item.get("size")
                    or sha256_file(path) != item.get("sha256")):
                problems.append("pre-change source entry is incomplete: " + str(relative)); break
        ownership = json.loads((archive / "source/pre-change-ownership.json").read_text())
        ownership_entries = ownership.get("entries") or []
        if (len(ownership_entries) != len(prechange["entries"])
                or any(not all(key in item for key in ("path", "uid", "gid"))
                       for item in ownership_entries)):
            problems.append("pre-change source ownership metadata is incomplete")
    if not any((archive / "evidence").rglob("*")):
        problems.append("read-only operational evidence is missing")
    nccl = archive / "artifacts/libnccl.so.2"
    if nccl.is_file() and (nccl.stat().st_size != EXPECTED_NCCL_SIZE or sha256_file(nccl) != EXPECTED_NCCL_SHA):
        problems.append("archived NCCL identity")
    if rerender and not problems:
        with tempfile.TemporaryDirectory(prefix="f0-verify-render-", dir="/private/tmp") as temp:
            generated = Path(temp) / "rendered"
            check = render_private(archive / "source/completed-iac",
                                   archive / "private/site/f0-cluster-resolved.env", generated)
            prior = json.loads((archive / "rendered/render.json").read_text())
            if check.get("status") != "PASS" or check.get("generated") != prior.get("generated"):
                problems.append("netplan render is not reproducible")
    return problems


def live_compare(archive: Path, args: argparse.Namespace,
                 collector: Callable = collect_rank) -> tuple[list[str], list[dict[str, Any]]]:
    recipe = json.loads((archive / "private/site/effective-recipe.json").read_text())
    hosts = recipe.get("hosts", "").split(); problems = []
    if len(hosts) != 4: return ["archive does not contain four hosts"], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(collector, rank, host, recipe, args.timeout,
                               args.command_limit, args.total_limit)
                   for rank, host in enumerate(hosts)]
        current = [future.result() for future in futures]
    state = json.loads((archive / "archive.json").read_text())
    captured = state.get("runtime_signatures") or []
    for rank, value in enumerate(current):
        prior_path = archive / f"ranks/rank-{rank}/receipt.json"
        prior = json.loads(prior_path.read_text()) if prior_path.is_file() else {}
        changed_sections = []
        for section in ("identity_after", "files", "model_availability", "loaded_nccl",
                        "managed_state", "host_state", "nvidia_identity"):
            before = prior.get(section)
            after = value.get(section)
            if section == "files":
                before = [{k: v for k, v in item.items()
                           if k not in ("archive_path", "content_base64")}
                          for item in before or []]
                after = [{k: v for k, v in item.items()
                          if k not in ("archive_path", "content_base64")}
                         for item in after or []]
            if before != after:
                detail: dict[str, Any] = {"section": section}
                if isinstance(before, dict) and isinstance(after, dict):
                    detail["changed_keys"] = sorted(
                        key for key in set(before) | set(after) if before.get(key) != after.get(key))
                elif section == "files":
                    old_files = {item.get("requested", item.get("path")): item for item in before}
                    new_files = {item.get("requested", item.get("path")): item for item in after}
                    detail["changed_keys"] = sorted(
                        str(key) for key in set(old_files) | set(new_files)
                        if old_files.get(key) != new_files.get(key))
                changed_sections.append(detail)
        value["comparison"] = {"captured_runtime_signature": captured[rank]
                               if rank < len(captured) else None,
                               "changed_sections": changed_sections}
        if value.get("capture_status") != "complete":
            problems.extend(f"rank {rank}: {item}" for item in value.get("problems") or ["live query incomplete"])
        if rank >= len(captured) or value.get("runtime_signature") != captured[rank]:
            names = ", ".join(item["section"] for item in changed_sections) or "signature only"
            problems.append(f"rank {rank}: runtime/configuration identity changed ({names})")
    return sorted(set(problems)), current


def make_report_root(requested: Path | None) -> Path:
    parent = outside_repo(requested or Path(tempfile.gettempdir()))
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="f0-reference-report-", dir=parent)); root.chmod(0o700)
    return root


def verify(args: argparse.Namespace, *, live: bool,
           collector: Callable = collect_rank) -> int:
    archive = outside_repo(args.archive)
    problems = validate_archive(archive)
    report: dict[str, Any] = {"schema": 1, "checked_at": utcnow(), "mode": "live" if live else "offline",
                              "archive": str(archive), "problems": problems}
    try:
        readiness = json.loads((archive / "archive.json").read_text()).get(
            "operational_readiness", {"status": "UNKNOWN", "reasons": []})
    except (OSError, json.JSONDecodeError):
        readiness = {"status": "UNKNOWN", "reasons": ["archive readiness record unreadable"]}
    report["operational_readiness"] = readiness
    directory = make_report_root(args.report_root)
    if live and not problems:
        live_problems, ranks = live_compare(archive, args, collector)
        problems.extend(live_problems)
        for rank, value in enumerate(ranks):
            write_json(directory / f"current/rank-{rank}.json", value)
        report["rank_status"] = [{"rank": rank, "capture_status": value.get("capture_status"),
                                  "runtime_signature": value.get("runtime_signature"),
                                  "comparison": value.get("comparison")}
                                 for rank, value in enumerate(ranks)]
    report["status"] = "PASS" if not problems else "FAIL"
    write_json(directory / "report.json", report)
    print(f"F0 REFERENCE {'LIVE' if live else 'OFFLINE'} {report['status']} "
          f"readiness={readiness.get('status','UNKNOWN')} report={directory / 'report.json'}")
    return 0 if not problems else 1


def materialize_source_tree(source: Path, records: list[dict[str, Any]], destination: Path) -> None:
    if destination.exists(): raise ReferenceError("restore staging destination already exists")
    shutil.copytree(source, destination)
    for current, dirs, _files in os.walk(destination):
        Path(current).chmod(0o700)
        for name in dirs: (Path(current) / name).chmod(0o700)
    for item in records:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or item.get("type") != "file":
            raise ReferenceError("unsafe completed IaC source metadata")
        path = destination / relative
        if (not path.is_file() or path.stat().st_size != item.get("size")
                or sha256_file(path) != item.get("sha256")):
            raise ReferenceError("completed IaC source hash mismatch")
        path.chmod(int(item["mode"]))


def stage_source(args: argparse.Namespace) -> int:
    archive = outside_repo(args.archive); destination = outside_repo(args.destination)
    problems = validate_archive(archive, rerender=True)
    if problems:
        print("F0 REFERENCE STAGE FAIL archive verification")
        return 1
    records = json.loads((archive / "source/completed-iac-manifest.json").read_text())
    try:
        materialize_source_tree(archive / "source/completed-iac", records, destination)
        cluster = destination / "cluster.env"
        shutil.copyfile(archive / "private/site/f0-cluster-resolved.env", cluster); cluster.chmod(0o600)
        syntax = subprocess.run(["bash", "-n", str(cluster)], capture_output=True, check=False)
        if syntax.returncode: raise ReferenceError("materialized F0 cluster.env has invalid shell syntax")
    except Exception:
        print("F0 REFERENCE STAGE FAIL source materialization")
        return 1
    print(f"F0 REFERENCE STAGE PASS destination={destination}")
    return 0


def archived_controller_source(archive: Path, expected_sha: str) -> Path:
    # Support sealed archives created before the reference controller was split out.
    # A present but changed frozen copy must not silently fall back to another file.
    for relative in ("scripts/node/reference/tp4ctl-f0-20260912.sh", "scripts/tp4ctl"):
        source = archive / "source/completed-iac" / relative
        if source.is_file():
            if sha256_file(source) != expected_sha:
                raise ReferenceError("archived Previous controller hash mismatch")
            return source
    raise ReferenceError("archived Previous controller missing")


def plan_restore(args: argparse.Namespace, collector: Callable = collect_rank) -> int:
    archive = outside_repo(args.archive)
    offline = validate_archive(archive)
    live, current = (live_compare(archive, args, collector) if not offline else ([], []))
    directory = make_report_root(args.report_root)
    for rank, value in enumerate(current):
        write_json(directory / f"current/rank-{rank}.json", value)
    recipe = json.loads((archive / "private/site/effective-recipe.json").read_text()) if not offline else {}
    state = json.loads((archive / "archive.json").read_text()) if not offline else {}
    comparison = offline + live
    rank_homes = []
    if not offline:
        for rank in range(4):
            receipt = json.loads((archive / f"ranks/rank-{rank}/receipt.json").read_text())
            installed = next((item for item in receipt.get("files") or []
                              if item.get("requested") == "$HOME/tp4/cluster.env"), {})
            rank_homes.append(str(Path(installed.get("path", "/missing/tp4/cluster.env")).parent.parent))
    deploy_user = Path(rank_homes[0]).name if rank_homes else "<unresolved>"
    ssh_targets = recipe.get("hosts", "").split()
    mgmt_ips = recipe.get("mgmt_ips", "").split()
    remote_mesh = " ".join(f"{deploy_user}@{address}" for address in mgmt_ips)
    current_overlay = args.current_overlay
    if current_overlay is None and not live:
        current_overlay = state.get("observed_tp4_env") or ""
    down_env = ([f"TP4_ENV={current_overlay}"] if current_overlay else []) + [f"TP4_HOSTS={remote_mesh}"]
    rank0 = ssh_targets[0] if ssh_targets else "<rank0>"
    rank1 = ssh_targets[1] if len(ssh_targets) > 1 else "<rank1>"
    rank0_home = rank_homes[0] if rank_homes else "/home/<USER>"
    rank1_home = rank_homes[1] if len(rank_homes) > 1 else "/home/<USER>"
    work = f"/private/tmp/f0-restore-{state.get('reference_id','f0-20260912')}"
    stage_controller = rank0_home + "/tp4/.tp4ctl-f0-reference.stage"
    controller = rank0_home + "/tp4/tp4ctl-f0-reference"
    controller_sha = next(item["sha256"] for item in json.loads(REFERENCE_MANIFEST.read_text())["artifacts"]
                          if item["path"] == "scripts/node/reference/tp4ctl-f0-20260912.sh")
    controller_source = archive / "source/completed-iac/scripts/node/reference/tp4ctl-f0-20260912.sh"
    controller_problem = ""
    if not offline:
        try:
            controller_source = archived_controller_source(archive, controller_sha)
        except ReferenceError as exc:
            controller_problem = str(exc)
            comparison.append(controller_problem)
    stage_nccl = rank1_home + "/tp4/f0-reference/libnccl.so.2"
    stage_dropin = rank0_home + "/tp4/.20-f0-reference.conf.stage"
    remote_install_controller = shlex.join([
        "install", "-m", "0700", stage_controller, controller,
    ]) + " && " + shlex.join(["sha256sum", controller])
    remote_down = shlex.join(["env", *down_env, controller, "down"])
    remote_dropin = (shlex.join(["sudo", "-n", "install", "-d", "-o", "root", "-g", "root", "-m", "0755",
                                  "/etc/systemd/system/tp4-autostart.service.d"])
                     + " && " +
                     shlex.join(["sudo", "-n", "install", "-o", "root", "-g", "root", "-m", "0644",
                                  stage_dropin, "/etc/systemd/system/tp4-autostart.service.d/20-f0-reference.conf"])
                     + " && " + shlex.join(["sudo", "-n", "systemctl", "daemon-reload"]))
    remote_up = shlex.join(["env", "TP4_ENV=scripts/node/reference/f0-20260912.env",
                             f"TP4_HOSTS={remote_mesh}", controller, "up"])
    commands = [
        (f"python3 {shlex.quote(str(archive / 'source/completed-iac/scripts/f0-reference.py'))} "
         f"stage-source --archive {shlex.quote(str(archive))} --destination {shlex.quote(work)}"),
        f"python3 {shlex.quote(work + '/scripts/f0-reference.py')} verify --offline --archive {shlex.quote(str(archive))}",
        shlex.join(["scp", "-p", str(controller_source), f"{rank0}:{stage_controller}"]),
        shlex.join(["ssh", rank0, remote_install_controller]) + f"  # expect {controller_sha}",
        shlex.join(["ssh", rank0, remote_down]),
        f"cd {shlex.quote(work)} && TP4_ENV=scripts/node/reference/f0-20260912.env bash scripts/deploy.sh --check",
        f"cd {shlex.quote(work)} && TP4_ENV=scripts/node/reference/f0-20260912.env bash scripts/deploy.sh",
        shlex.join(["ssh", rank1, "mkdir -p " + shlex.quote(rank1_home + "/tp4/f0-reference")]),
        shlex.join(["scp", "-p", str(archive / "artifacts/libnccl.so.2"), f"{rank1}:{stage_nccl}"]),
        (f"cd {shlex.quote(work)} && bash scripts/node/nccl/install-nccl.sh "
         f"--from {shlex.quote(rank1 + ':' + stage_nccl)} --expect-sha {EXPECTED_NCCL_SHA}"),
        shlex.join(["scp", "-p", str(archive / "private/prepared/tp4-autostart.service.d/20-f0-reference.conf"),
                    f"{rank0}:{stage_dropin}"]),
        shlex.join(["ssh", rank0, remote_dropin]),
        shlex.join(["ssh", rank0, remote_up]),
        shlex.join(["ssh", rank0, f"curl -fsS http://localhost:{recipe.get('api_port','8000')}/health"]),
        f"cd {shlex.quote(work)} && TP4_ENV=scripts/node/reference/f0-20260912.env python3 scripts/check-f0.py --baseline docs/baseline-f0.json",
    ]
    if controller_problem:
        commands[2] = "BLOCKED: " + controller_problem
    if live and args.current_overlay is None:
        commands[4] = "BLOCKED: live identity differs; supply --current-overlay for the currently running recipe before generating a down command"
    document = f"""# F0 restore review plan

Generated {utcnow()}. No restore command in this document was executed. The current
comparison used read-only SSH when archive integrity allowed it.

Archive: {archive}
Current comparison: {'no differences detected' if not comparison else '; '.join(comparison)}
Captured operational readiness: {(state.get('operational_readiness') or {}).get('status','UNKNOWN')}

## Runtime recipe (future approved four-rank window)

1. Verify the archive offline and live. Resolve every difference and prove all four ranks reachable.
2. Stage the archived controller as a regular file on rank 0; verify SHA-256. Never embed its source in `bash -c`.
3. Use the currently running overlay for one coordinated four-rank `down`; reference F0 is selected only after the old process is stopped.
4. Deploy the completed archived IaC and exact archived NCCL bytes through the repository's additive deploy/install procedures; verify every destination hash.
5. Install the prepared autostart drop-in only after review, then daemon-reload. This selects the frozen overlay for the next boot.
6. Run fabric prerequisites, one coordinated four-rank `up`, wait for `/health` 200, run both functional gates within 120 seconds, then `scripts/check-f0.py --baseline docs/baseline-f0.json`.

## Host and fabric state (deferred review, never automatic)

The runtime procedure does not apply netplan, flush firewall rules, activate NetworkManager
profiles, change devlink/eSwitch/TC/offloads, packages, driver, firmware, kernel, IOMMU, GRUB,
boot parameters or reboot a node. Preserve management, Tailscale and SSH recovery paths and
unrelated host configuration. Restore only repository-owned files after comparing each target.
Host/network/boot work requires a separate authorized window and verified independent recovery
access or the owner's physical availability. If a reboot is approved, rank 0 is last.

## Exact runtime commands for review

```sh
{os.linesep.join(commands)}
```

Commands from the first `scp` onward mutate remote state. The coordinated down/up, deploy,
NCCL installation, autostart drop-in, daemon-reload and functional requests require an
approved maintenance window. The `/health` command is only readiness; run the two exact
functional requests in `docs/operations.md` within 120 seconds of its first 200 response.

Targets in the private recipe: {len(recipe.get('hosts','').split())}. Review the private
per-rank receipts for exact paths, modes, ownership, symlinks and loaded-vs-disk unit state.
"""
    write_private(directory / "restore-plan.md", document.encode())
    write_json(directory / "comparison.json", {"schema": 1, "created_at": utcnow(),
               "offline_problems": offline, "live_differences": live,
               "controller_problem": controller_problem,
               "rank_status": [{"rank": i, "capture_status": item.get("capture_status"),
                                 "comparison": item.get("comparison")}
                               for i, item in enumerate(current)],
               "read_only_comparison_executed": not bool(offline),
               "restore_commands_executed": False})
    print(f"F0 REFERENCE RESTORE PLAN report={directory / 'restore-plan.md'}")
    return 0 if not comparison else 1


def bounded_positive(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0 or parsed > 600:
        raise argparse.ArgumentTypeError("timeout must be finite and between 0 and 600 seconds")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--archive", required=True, type=Path)
    shared.add_argument("--timeout", type=bounded_positive, default=180.0)
    shared.add_argument("--command-limit", type=int, default=2 * 1024 * 1024, help=argparse.SUPPRESS)
    shared.add_argument("--total-limit", type=int, default=16 * 1024 * 1024, help=argparse.SUPPRESS)
    shared.add_argument("--report-root", type=Path, help=argparse.SUPPRESS)
    cap = sub.add_parser("capture", parents=[shared], help="capture a new private reference")
    cap.add_argument("--prechange-source", type=Path, required=True)
    cap.add_argument("--evidence-dir", type=Path, action="append", required=True)
    cap.add_argument("--nccl-timeout", type=bounded_positive, default=300.0)
    verify_parser = sub.add_parser("verify", help="verify a captured reference")
    verify_parser.add_argument("--offline", action="store_true")
    verify_parser.add_argument("--live", action="store_true")
    verify_parser.add_argument("--archive", required=True, type=Path)
    verify_parser.add_argument("--timeout", type=bounded_positive, default=180.0)
    verify_parser.add_argument("--command-limit", type=int, default=2 * 1024 * 1024, help=argparse.SUPPRESS)
    verify_parser.add_argument("--total-limit", type=int, default=16 * 1024 * 1024, help=argparse.SUPPRESS)
    verify_parser.add_argument("--report-root", type=Path, help=argparse.SUPPRESS)
    plan = sub.add_parser("plan-restore", parents=[shared], help="write a non-executing restore plan")
    plan.add_argument("--current-overlay", help="relative TP4_ENV of the currently running stack")
    stage = sub.add_parser("stage-source", help="materialize a verified private restore source tree")
    stage.add_argument("--archive", required=True, type=Path)
    stage.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "verify" and args.offline == args.live:
        parser.error("verify requires exactly one of --offline or --live")
    if args.command in ("capture", "verify", "plan-restore") and (
            args.command_limit < 1024 or args.total_limit < 4096):
        parser.error("capture bounds are too small")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "capture": return capture(args)
    if args.command == "verify": return verify(args, live=args.live)
    if args.command == "plan-restore": return plan_restore(args)
    if args.command == "stage-source": return stage_source(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
