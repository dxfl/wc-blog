
#k-nearest neighbors from "programming collective intelligence" book

module Knn

  def self.euclidean(p,q)
    d = 0.0
    p.size.times do |i|
      d += (p[i] - q[i])**2
    end
    d**0.5
  end

  def self.getdistances(data,vec1)
    distancelist = []
    data.size.times do |i|
      vec2 = data[i][:features]
      distancelist << [euclidean(vec1, vec2), i]
    end
    distancelist.sort
  end

  def self.gaussian(dist, sigma=10.0)
    Math.exp(-dist**2/(2*sigma**2))
  end

  # simple knn without weighted interactions
  def self.knnestimate(data, vec1, k=5)
    distancelist = getdistances(data, vec1[:features])
    avg = 0.0
    0.upto(k-1) do |i|
      point_id = distancelist[i][1]
      avg += data[point_id][:label]
    end
    avg = avg/k
  end

  # weighted knn
  def self.weightedknn(data, vec1, k=5)
    dlist = getdistances(data, vec1[:features])
    avg = 0.0
    totalweight = 0.0
    0.upto(k-1) do |i|
      dist = dlist[i][0]
      point_id = dlist[i][1]
      weight = gaussian(dist)
      avg += weight*data[point_id][:label]
      totalweight += weight
    end
    avg = avg/totalweight
  end

  def self.maxymin(data)
    max = Array.new(data[0][:features].size, 0)
    min = Array.new(data[0][:features].size, 0)
    data.each do |vect|
      vect[:features].each_index do |i|
        max[i] = vect[:features][i].to_f if max[i] < vect[:features][i]
        min[i] = vect[:features][i].to_f if min[i] > vect[:features][i]
      end
    end
    return max, min
  end

  def self.normalize(data, point)
    all_data = data + [point]
    max,min = maxymin(all_data)
    rescaling(data,max,min)
    rescaling([point],max,min)
  end

  def self.rescaling(data,max,min)
    data.each do |vect|
      vect[:features].each_index{ |i| vect[:features][i] = (vect[:features][i] - min[i]) / (max[i] - min[i]) }
    end
  end
  
end
