import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def main():
    basevars = ["dynamic_range", "visibilities_fraction"]
    metricvars = ["metrics.computation_time_seconds.dirty_image",
                  "metrics.computation_time_seconds.psf",
                  "metrics.computation_time_seconds.clean_image",
                  "metrics.computation_time_seconds.total",
                  "metrics.memory_consumption_megabytes.dirty_image",
                  "metrics.memory_consumption_megabytes.psf",
                  "metrics.memory_consumption_megabytes.clean_image",
                  "metrics.memory_consumption_megabytes.total",
                  "metrics.image_metrics.cross_correlation",
                  "metrics.image_metrics.snr",
                  "metrics.image_metrics.psnr",
                  "metrics.image_metrics.flux_ratio",
                  "metrics.image_metrics.residual_rms",
                  "metrics.image_metrics.residual_rms_over_sigma",
                  "metrics.image_metrics.achieved_dr"]

    text = ""
    # for base in basevars:
    #     for metric in metricvars:
    #         corr = correlation(base, metric)

    #         newbase = base.replace("_", "\\_")
    #         newmetric = metric.replace("_", "\\_")

    #         text += f"\\item Correlation between {newbase} and {newmetric} is ${corr}$\n"
    #     text += "\n"

    for metric in metricvars:
        text += "\\textbf{" + metric + "}"
        for base in basevars:
            text += f"& ${correlation(metric, base):.5f}$ "
        text += "\\\\\n"
    
    with open("latex-copypaste.txt", "w") as fd:
        fd.write(text)



def correlation(var1, var2):
    df = pd.read_csv("./metrics_summary.csv")

    var1_array = np.nan_to_num(np.array(df[var1]))
    var2_array = np.nan_to_num(np.array(df[var2]))
    length = len(var1_array)

    var1_avg = np.mean(var1_array)
    var2_avg = np.mean(var2_array)

    num = 0
    denom1 = 0
    denom2 = 0
    for i in range(length):
        num += (var1_array[i] - var1_avg) * (var2_array[i] - var2_avg)
        denom1 += (var1_array[i] - var1_avg) ** 2
        denom2 += (var2_array[i] - var2_avg) ** 2

    corr = num / np.sqrt(denom1 * denom2)
    return corr



if __name__ == "__main__":
    main()