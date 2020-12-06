function log_header(verbose, file, args...)
   if verbose >= 2
      for param in args
         if isa(param, Pair)
            key = param[1]
            value = param[2]
            @printf("SADDLESEARCH:%18s  =  %1.2e\n", key, value)
         end
      end
      @printf("SADDLESEARCH: ------------------------------\n")
      @printf("SADDLESEARCH:  time | nit |    sup|∇E|_∞    \n")
      @printf("SADDLESEARCH: ------|-----|-----------------\n")
   end

   if verbose >= 4 && file!=nothing
      for param in args
         if isa(param, Pair)
            key = param[1]
            value = param[2]
            strlog = @sprintf("SADDLESEARCH:%18s  =  %1.2e\n", key, value)
            write(file, strlog)
            flush(file)
         end
      end
      overline = @sprintf("SADDLESEARCH: ------------------------------\n")
      strlog = @sprintf("SADDLESEARCH:  time | nit |    sup|∇E|_∞ \n")
      underline = @sprintf("SADDLESEARCH: ------|-----|-----------------\n")
      write(file, overline, strlog, underline)
      flush(file)
   end
end

function log_history(verbose, file, nit, Rn)
   if verbose >= 2
      dt = Dates.format(now(), "HH:MM")
      @printf("SADDLESEARCH: %s |%4d |     %1.2e\n", dt, nit, Rn)
   end
   if verbose >= 4 && file!=nothing
      dt = Dates.format(now(), "HH:MM")
      strlog = @sprintf("SADDLESEARCH: %s |%4d |     %1.2e\n", dt, nit, Rn)
      write(file, strlog)
      flush(file)
   end
end

function warn_maxtol(verbose, file, nit, Rn)
   if verbose >= 4 && file!=nothing
      strlog = @sprintf("SADDLESEARCH: Residual %s too large at nit = %s.\n", "$Rn", "$nit")
      write(file, strlog)
      close(file)
   end
   @warn("SADDLESEARCH: Residual $Rn is too large at nit = $nit.");
end

function log_acceptstep(verbose, file, h, h_ls, h_err, Rn)
   if verbose >= 3
      @printf("SADDLESEARCH: %30s accept: new h =%15.11f\n", "|", h)
      @printf("SADDLESEARCH: %30s           |F| =%15.11f\n", "|", Rn)
      @printf("SADDLESEARCH: %30s           hls =%15.11f\n", "|", h_ls)
      @printf("SADDLESEARCH: %30s          herr =%15.11f\n", "|", h_err)
   end
   if verbose >= 4 && file!=nothing
      str_h = @sprintf("SADDLESEARCH: %30s accept: new h =%15.11f\n","|",  h)
      str_F = @sprintf("SADDLESEARCH: %30s           |F| =%15.11f\n", "|", Rn)
      str_hls = @sprintf("SADDLESEARCH: %30s           hls =%15.11f\n", "|", h_ls)
      str_herr = @sprintf("SADDLESEARCH: %30s          herr =%15.11f\n", "|", h_err)
      write(file, str_h, str_F, str_hls, str_herr)
      flush(file)
   end
end

function log_rejectstep(verbose, file, h, Rnew, Rn)
   if verbose >= 3
      @printf("SADDLESEARCH: %30s reject: new h =%15.11f\n", "|", h)
      @printf("SADDLESEARCH: %30s        |Fnew| =%15.11f\n", "|", Rnew)
      @printf("SADDLESEARCH: %30s        |Fold| =%15.11f\n", "|", Rn)
      @printf("SADDLESEARCH: %30s |Fnew|/|Fold| =%15.11f\n", "|", (Rnew/Rn))
   end
   if verbose >= 4 && file!=nothing
      str_h = @sprintf("SADDLESEARCH: %30s reject: new h =%15.11f\n", "|", h)
      str_Fnew = @sprintf("SADDLESEARCH: %30s        |Fnew| =%15.11f\n", "|", Rnew)
      str_Fold = @sprintf("SADDLESEARCH: %30s        |Fold| =%15.11f\n", "|", Rn)
      str_Fratio = @sprintf("SADDLESEARCH: %30s |Fnew|/|Fold| =%15.11f\n", "|", (Rnew/Rn))
      write(file, str_h, str_Fnew, str_Fold, str_Fratio)
      flush(file)
   end
end

function warn_hmin(verbose, file, h, nit)
   if verbose >= 4 && file!=nothing
      strlog = @sprintf("SADDLESEARCH: Step size %s too small at nit = %s.\n", "$h", "$nit")
      write(file, strlog)
      close(file)
   end
   @warn("SADDLESEARCH: Step size $h too small at nit = $nit.");
end

function error_parameter(verbose, file, param)
   if verbose >= 4 && file!=nothing
      strlog = @sprintf("SADDLESEARCH: invalid %s parameter.\n", param)
      write(file, strlog)
      close(file)
   end
   error("SADDLESEARCH: invalid $(param) parameter.")
end

function log_converged(verbose, file, method, nit)
   # tail = "s."
   if verbose >= 1
      println("SADDLESEARCH: $(method) terminates succesfully after $(nit) iterations.")
   end
   if verbose >= 4 && file!=nothing
      strlog = @sprintf("SADDLESEARCH: %s terminates succesfully after %s iterations.\n", "$(method)", "$(nit)")
      write(file, strlog)
      close(file)
   end
end

function log_diverged(verbose, file, method, maxnit)
   if verbose >= 1
      println("SADDLESEARCH: $(method) terminated unsuccesfully after $(maxnit) iterations.")
   end
   if verbose >= 4 && file!=nothing
      strlog = @sprintf("SADDLESEARCH: %s terminated unsuccesfully after %s iterations.\n", "$(method)", "$(maxnit)")
      write(file, strlog)
   end

   if verbose >= 4 && file!=nothing
      close(file)
   end
end
