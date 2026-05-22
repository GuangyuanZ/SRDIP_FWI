# SRDIP_FWI

## Installation

Before running this project, please first install both **SWIT-1.0** and **ADFWI**.

---

### Step 1. Install SWIT-1.0

Please follow the installation instructions of SWIT-1.0:

https://github.com/seisfwi/SWIT

---

### Step 2. Install ADFWI

Please follow the installation instructions of ADFWI:

https://github.com/liufeng2317/ADFWI

---

### Step 3. Replace SWIT Components

After installing SWIT-1.0, replace the following directories/files in the SWIT folder using the contents provided in `SRDIP_FWI`:

```text
SRDIP_FWI/toolbox   -->   SWIT/toolbox
SRDIP_FWI/fd2dmpi  -->   SWIT/fd2dmpi
```

### Step 4. Recompile fd2dmpi

After replacing `fd2dmpi`, recompile the solver:

```bash
cd .../fd2dmpi
make clean
make
```
