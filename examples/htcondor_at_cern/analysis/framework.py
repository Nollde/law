# coding: utf-8

"""
Law example tasks to demonstrate HTCondor workflows at CERN.

In this file, some really basic tasks are defined that can be inherited by
other tasks to receive the same features. This is usually called "framework"
and only needs to be defined once per user / group / etc.
"""


import os

import luigi
import law


# the htcondor workflow implementation is part of a law contrib package
# so we need to explicitly load it
law.contrib.load("htcondor")


class Task(law.Task):
    """
    Base task that we use to force a version parameter on all inheriting tasks, and that provides
    some convenience methods to create local file and directory targets at the default data path.
    """

    version = luigi.Parameter()

    def store_parts(self):
        return (self.__class__.__name__, self.version)

    def local_path(self, *path):
        # ANALYSIS_DATA_PATH is defined in setup.sh
        parts = (os.getenv("ANALYSIS_DATA_PATH"),) + self.store_parts() + path
        return os.path.join(*parts)

    def local_target(self, *path):
        return law.LocalFileTarget(self.local_path(*path))


class HTCondorWorkflow(law.HTCondorWorkflow):
    """
    Batch systems are typically very heterogeneous by design, and so is HTCondor. Law does not aim
    to "magically" adapt to all possible HTCondor setups which would certainly end in a mess.
    Therefore we have to configure the base HTCondor workflow in law.contrib.htcondor to work with
    the CERN HTCondor environment. In most cases, like in this example, only a minimal amount of
    configuration is required.
    """

    # internally, the job files required for htcondor submission are created in a "job file factory"
    # here, we can configure the factory to store the job files for subsequent inspection
    htcondor_job_file_factory_defaults = {"is_tmp": False}

    def htcondor_output_directory(self):
        # the directory where submission meta data should be stored
        return law.LocalDirectoryTarget(self.local_path())

    def htcondor_bootstrap_file(self):
        # each job can define a bootstrap file that is executed prior to the actual job
        # in order to setup software and environment variables
        return law.util.rel_path(__file__, "bootstrap.sh")

    def htcondor_job_config(self, config, job_num, branches):
        # render_data is rendered into all files sent with a job
        config.render_variables["analysis_path"] = os.getenv("ANALYSIS_PATH")
        # copy the entire environment
        config.custom_content.append(("getenv", "true"))
        # the CERN htcondor setup requires a "log" config, but we can safely set it to /dev/null
        # if you are interested in the logs of the batch system itself, set a meaningful value here
        config.custom_content.append(("log", "/dev/null"))
        return config
