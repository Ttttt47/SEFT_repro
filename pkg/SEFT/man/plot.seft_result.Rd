\name{plot.seft_result}
\alias{plot.seft_result}
\title{Plot regional SEFT results}
\usage{
\method{plot}{seft_result}(x, file = NULL, tfce_coverage = NULL, main = NULL, ...)
}
\arguments{
\item{x}{A \code{seft_result} object.}
\item{file}{Optional PDF filename.}
\item{tfce_coverage}{Optional data frame with \code{region_id} and \code{tfce_coverage} columns.}
\item{main}{Optional title.}
\item{...}{Unused.}
}
\value{The input object, invisibly.}
\description{Displays the highest significant partial-conjunction level for each region and each available method.}
