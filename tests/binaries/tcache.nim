## Cache layout, REW_CACHE_DIR override, and version isolation.

import std/[os, strutils]
import rew/binaries/[cache, manifest, target]

block cache_dir_respects_env:
  let old = getEnv("REW_CACHE_DIR")
  putEnv("REW_CACHE_DIR", "/tmp/rew_test_cache_dir")
  doAssert baseCacheDir() == "/tmp/rew_test_cache_dir"
  if old.len > 0:
    putEnv("REW_CACHE_DIR", old)
  else:
    delEnv("REW_CACHE_DIR")

block versioned_dir_includes_version:
  let v = manifest().rewVersion
  doAssert v in versionedCacheDir()

block subdirs_are_correct:
  doAssert downloadDir().endsWith("download")
  doAssert externalDir().endsWith("external")
  doAssert buildDir().endsWith("build")
  doAssert pluginsDir().endsWith("plugins")
  doAssert executablesDir().endsWith("executables")

block plugin_path_for_target:
  let p = pluginPathForTarget(tCpu)
  doAssert "pjrt_c_api_cpu_plugin" in p

echo "tcache: OK"
