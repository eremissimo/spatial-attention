module WindowFunctions

    export AbstractWindow, MeanWindow, SumWindow, GaussianWindow, get_radius, construct_kernel

    abstract type AbstractWindow end

    struct MeanWindow <: AbstractWindow
        radius
        val
    end

    """
        SumWindow(radius)

    Constructs a sum filter (sum of values at each pixel when convolved).
    """
    struct SumWindow <: AbstractWindow
        radius
    end

    struct GaussianWindow <: AbstractWindow
        radius
        sigma
    end

    """
        MeanWindow(radius [, val=1/(2*radius+1)^2])

    Constructs a mean filter (i.e. convolution with it computes the mean value of
    a neighborhood at each pixel).
    """
    function MeanWindow(radius)
        val=1/(2*radius+1)^2
        return MeanWindow(radius, val)
    end

    """
        GaussianWindow(radius [, sigma=radius/1.5])

    Constructs a gaussian blur filter.
    """
    function GaussianWindow(radius) where {T, N}
        sigma = radius/1.5                 # ¯\_(ツ)_/¯
        return GaussianWindow(radius, sigma)
    end

    get_radius(self::AbstractWindow) = self.radius

    function construct_kernel(self::MeanWindow, nd, type)
        return fill(convert(type, self.val), ntuple(i->(2*self.radius+1), nd))
    end

    function construct_kernel(self::SumWindow, nd, type)
        return ones(type, ntuple(i->(2*self.radius+1), nd))
    end

    function construct_kernel(self::GaussianWindow, nd, type)
        shape = ntuple(i->(2*self.radius+1), nd)
        s2 = self.sigma^2
        output = Array{type, nd}(undef, shape...)
        center = CartesianIndex(ntuple(i -> self.radius + 1, nd))
        for idx in CartesianIndices(output)
            output[idx] = exp(-0.5*sum(Tuple(idx - center).^2)/s2)
        end
        normalizer = sum(output)
        output ./= normalizer
        return output
    end


end    #module
