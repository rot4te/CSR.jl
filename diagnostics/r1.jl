using CSR, GTPSA
d = Descriptor(2, 2); x = @vars(d)
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
geom0 = CSRGeometry([0.2, 0.3, 0.2], [0.0, 1.5, 0.0], c3)
@show typeof(geom0)
try
  CSRKickPoint(geom0, 2, 0.17, 200.0 + x[1])
catch e
  Base.showerror(stdout, e, catch_backtrace()); println()
end
