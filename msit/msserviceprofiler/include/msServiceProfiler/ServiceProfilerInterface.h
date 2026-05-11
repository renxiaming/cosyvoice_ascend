/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MS_SERVER_PROFILER_INTERFACE_H
#define MS_SERVER_PROFILER_INTERFACE_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <set>
#include <string>
#include <cstdlib>
#include <sys/stat.h>
#include <linux/limits.h>

using SpanHandle = uint64_t;

#define MS_SERVICE_PROFILER_API __attribute__((visibility("default")))
#define MS_SERVICE_PROFILER_HIDDEN __attribute__((visibility("hidden")))

extern "C" {
MS_SERVICE_PROFILER_API SpanHandle StartSpan();
MS_SERVICE_PROFILER_API SpanHandle StartSpanWithName(const char *name);
MS_SERVICE_PROFILER_API void MarkSpanAttr(const char *msg, SpanHandle spanHandle);
MS_SERVICE_PROFILER_API void EndSpan(SpanHandle spanHandle);
MS_SERVICE_PROFILER_API void MarkEvent(const char *msg);
MS_SERVICE_PROFILER_API void StartServerProfiler();
MS_SERVICE_PROFILER_API void StopServerProfiler();
MS_SERVICE_PROFILER_API bool IsEnable(uint32_t level);
MS_SERVICE_PROFILER_API bool IsValidDomain(const char *domainName);
MS_SERVICE_PROFILER_API void AddMetaInfo(const char *key, const char *value);
}

namespace msServiceProfilerCompatible {
    class ServiceProfilerInterface {
    public:
        ServiceProfilerInterface(const ServiceProfilerInterface &) = delete;

        ServiceProfilerInterface &operator=(const ServiceProfilerInterface &) = delete;

        ServiceProfilerInterface(ServiceProfilerInterface &&) = delete;

        ServiceProfilerInterface &operator=(ServiceProfilerInterface &&) = delete;

    public:
        MS_SERVICE_PROFILER_HIDDEN static ServiceProfilerInterface &GetInstance()
        {
            static ServiceProfilerInterface logManager;
            return logManager;
        }

        ~ServiceProfilerInterface() = default;

        MS_SERVICE_PROFILER_HIDDEN inline SpanHandle CallStartSpanWithName(const char *name) const
        {
            return ptrStartSpanWithName_ ? ptrStartSpanWithName_(name) : 0;
        }

        MS_SERVICE_PROFILER_HIDDEN inline void CallMarkSpanAttr(const char *msg, SpanHandle spanHandle) const
        {
            if (ptrMarkSpanAttr_) {
                ptrMarkSpanAttr_(msg, spanHandle);
            }
        }

        MS_SERVICE_PROFILER_HIDDEN inline void CallEndSpan(SpanHandle spanHandle) const
        {
            if (ptrEndSpan_) {
                ptrEndSpan_(spanHandle);
            }
        }

        MS_SERVICE_PROFILER_HIDDEN inline void CallMarkEvent(const char *msg) const
        {
            if (ptrMarkEvent_) {
                ptrMarkEvent_(msg);
            }
        }

        MS_SERVICE_PROFILER_HIDDEN inline bool CallIsEnable(uint32_t level) const
        {
            return ptrIsEnable_ && ptrIsEnable_(level);
        }

        MS_SERVICE_PROFILER_HIDDEN inline bool CallIsDomainEnable(const char *currentDomain) const
        {
            if (ptrIsValidDomain_) {
                return ptrIsValidDomain_(currentDomain);
            } else {
                return true;
            }
        }

        MS_SERVICE_PROFILER_HIDDEN inline void CallStartServerProfiler() const
        {
            if (ptrStartServerProfiler_) {
                ptrStartServerProfiler_();
            }
        }

        MS_SERVICE_PROFILER_HIDDEN inline void CallStopServerProfiler() const
        {
            if (ptrStopServerProfiler_) {
                ptrStopServerProfiler_();
            }
        }

        MS_SERVICE_PROFILER_HIDDEN  inline void CallAddMetaInfo(const char *key, const char *value) const
        {
            if (ptrAddMetaInfo_) {
                ptrAddMetaInfo_(key, value);
            }
        }

    private:
        ServiceProfilerInterface()
        {
            char *ascendHomePathPtr = getenv("ASCEND_HOME_PATH");
            if (ascendHomePathPtr == nullptr) {
                printf("Get ASCEND_HOME_PATH failed. Please check that the CANN package is installed.\n"
                        "Run 'source set_env.sh' in the CANN installation path.\n");
                return;
            }
            std::string ascendHomePath(ascendHomePathPtr);
            if (ascendHomePath.empty()) {
                printf("Get ASCEND_HOME_PATH failed. Please check that the CANN package is installed.\n"
                        "Run 'source set_env.sh' in the CANN installation path.\n");
                return;
            }
            char ascendHomeRealPath[PATH_MAX + 1] = {0};
            if (realpath(ascendHomePath.c_str(), ascendHomeRealPath) == nullptr) {
                printf("Failed to canonicalize path: %s", strerror(errno));
                return;
            }
            std::string soName = std::string(ascendHomeRealPath) + "/lib64/libms_service_profiler.so";
            struct stat fileStat;
            if ((stat(soName.c_str(), &fileStat) != 0) || (fileStat.st_mode & S_IRUSR) == 0) {
                printf("File not readable: %s", soName.c_str());
                return;
            }
            auto handle = dlopen(soName.c_str(), RTLD_LAZY);
            if (handle) {
                ptrIsEnable_ = (decltype(IsEnable)*)dlsym(handle, "IsEnable");
                ptrStartSpanWithName_ = (decltype(StartSpanWithName)*)dlsym(handle, "StartSpanWithName");
                ptrMarkSpanAttr_ = (decltype(MarkSpanAttr)*)dlsym(handle, "MarkSpanAttr");
                ptrEndSpan_ = (decltype(EndSpan)*)dlsym(handle, "EndSpan");
                ptrMarkEvent_ = (decltype(MarkEvent)*)dlsym(handle, "MarkEvent");
                ptrStartServerProfiler_ = (decltype(StartServerProfiler)*)dlsym(handle, "StartServerProfiler");
                ptrStopServerProfiler_ = (decltype(StopServerProfiler)*)dlsym(handle, "StopServerProfiler");
                ptrIsValidDomain_ = (decltype(IsValidDomain)*)dlsym(handle, "IsValidDomain");
                ptrAddMetaInfo_ = (decltype(AddMetaInfo)*)dlsym(handle, "AddMetaInfo");
            }
        };

    private:
        decltype(IsEnable)* ptrIsEnable_ = nullptr;
        decltype(StartSpanWithName)* ptrStartSpanWithName_ = nullptr;
        decltype(MarkSpanAttr)* ptrMarkSpanAttr_ = nullptr;
        decltype(EndSpan)* ptrEndSpan_ = nullptr;
        decltype(MarkEvent)* ptrMarkEvent_ = nullptr;
        decltype(StartServerProfiler)* ptrStartServerProfiler_ = nullptr;
        decltype(StopServerProfiler)* ptrStopServerProfiler_ = nullptr;
        decltype(IsValidDomain)* ptrIsValidDomain_ = nullptr;
        decltype(AddMetaInfo)* ptrAddMetaInfo_ = nullptr;
    };
}

namespace msServiceProfiler {
    enum Level : uint32_t {
        ERROR = 10,
        INFO = 20,
        DETAILED = 30,
        VERBOSE = 40,
        LEVEL_CORE_TRACE = 10,
        LEVEL_OUTLIER_EVENT = 10,
        LEVEL_NORMAL_TRACE = 20,
        LEVEL_DETAILED_TRACE = 30,
        L0 = 10,
        L1 = 20,
        L2 = 30
    };
}  // namespace msServiceProfiler

#endif
