test_that("optional adapters share the standard working-model contract", {
    skip_on_os("windows")
    skip_if_not_installed("jsonlite")
    python <- Sys.which("python3")
    skip_if(!nzchar(python), "python3 is unavailable")
    root <- tempfile("seft-optional-fixture-")
    dir.create(file.path(root, "env", "bin"), recursive = TRUE)
    dir.create(file.path(root, "python"), recursive = TRUE)
    dir.create(file.path(root, "vendor"), recursive = TRUE)
    expect_true(file.symlink(python, file.path(root, "env", "bin", "python")))
    backend <- c(
        "#!/usr/bin/env python3",
        "import argparse, json, math, struct",
        "p=argparse.ArgumentParser()",
        "p.add_argument('--dims'); p.add_argument('--out-log-r'); p.add_argument('--out-log-rtil')",
        "p.add_argument('--out-r'); p.add_argument('--out-rtil'); p.add_argument('--diagnostics')",
        "a,_=p.parse_known_args(); dims=tuple(map(int,a.dims.split(','))); n=math.prod(dims)",
        "r=[0.2+0.6*i/max(1,n-1) for i in range(n)]; rt=list(reversed(r)); lr=list(map(math.log,r)); lt=list(map(math.log,rt))",
        "def write(path,values): open(path,'wb').write(struct.pack('<%sd'%len(values),*values))",
        "write(a.out_log_r,lr); write(a.out_log_rtil,lt)",
        "if a.out_r: write(a.out_r,r)",
        "if a.out_rtil: write(a.out_rtil,rt)",
        "json.dump({'fixture':True},open(a.diagnostics,'w'))"
    )
    writeLines(backend, file.path(root, "python", "fdr_smoothing_absmax.py"))
    writeLines(backend, file.path(root, "python", "ml_working_model.py"))
    old <- Sys.getenv("SEFT_OPTIONAL_ROOT", unset = NA_character_)
    on.exit(if (is.na(old)) Sys.unsetenv("SEFT_OPTIONAL_ROOT") else Sys.setenv(SEFT_OPTIONAL_ROOT = old), add = TRUE)
    Sys.setenv(SEFT_OPTIONAL_ROOT = root)
    fixture <- tiny_seft_fixture(c(8L, 8L, 8L))
    for (model in c("fdr-smoothing", "deepfdr", "fchmrf")) {
        result <- SEFT:::.run_working_model(
            fixture$z, fixture$z + 0.2, fixture$mask, model,
            seed = 1L, density_bandwidth = 1, bandwidth = 1,
            neighbor_range = 1L, lambda = 0.5, score_clip = 0.99,
            verbose = FALSE
        )
        expect_identical(dim(result$R), dim(fixture$z))
        expect_identical(dim(result$R_til), dim(fixture$z))
        expect_true(all(is.finite(result$log_R)))
        expect_true(all(is.finite(result$log_R_til)))
        expect_identical(result$model_identifier, paste0("seft_", model))
    }
})
